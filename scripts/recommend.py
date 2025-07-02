# scripts/recommend.py

import os
import json
import arxiv
import shutil
import time
import requests
import numpy as np
import re
import torch
from datetime import datetime
from collections import defaultdict
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util
from jinja2 import Environment, FileSystemLoader
from config import (
    ARXIV_CATEGORIES, EMBEDDING_MODEL, TASTE_PROFILE_PATH,
    RECOMMENDATION_LIMIT, LLM_RPM_LIMIT, DMRG_URL, DMRG_SOURCE_TAG
)

ARCHIVE_PATH = 'archive.json'
FOLLOWED_AUTHORS_PATH = 'followed_authors.json'
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
LLM_FAILURE_REASON = "Could not be analyzed by LLM."

def filter_arxiv_categories(categories):
    """Filters a list of strings to only include valid arXiv category formats."""
    if not categories:
        return []
    arxiv_category_pattern = re.compile(r'^[a-z\-]+\.[A-Za-z\-]+$')
    return [cat for cat in categories if arxiv_category_pattern.match(cat)]

def get_recent_papers():
    """Fetches the most recent papers from specified arXiv categories."""
    print(f"🔍 Fetching recent papers from categories: {', '.join(ARXIV_CATEGORIES)}")
    all_papers = {}
    client = arxiv.Client()
    for category in ARXIV_CATEGORIES:
        print(f"Querying category '{category}'...")
        search = arxiv.Search(query=f"cat:{category}", max_results=150, sort_by=arxiv.SortCriterion.SubmittedDate)
        try:
            for paper in client.results(search):
                paper_id = paper.entry_id.split('/abs/')[-1]
                if paper_id not in all_papers:
                    all_papers[paper_id] = paper
        except Exception as e:
            print(f"🔴 Error fetching from {category}: {e}")
    print(f"✅ Found a total of {len(all_papers)} unique recent papers.")
    return list(all_papers.values())

def get_papers_from_followed_authors():
    """Fetches recent papers from authors in followed_authors.json."""
    try:
        with open(FOLLOWED_AUTHORS_PATH, 'r') as f:
            followed_authors = json.load(f)
    except FileNotFoundError:
        return []

    if not followed_authors:
        return []

    print(f"🔍 Fetching papers from {len(followed_authors)} followed author(s)...")
    author_papers = {}
    client = arxiv.Client()
    for author in followed_authors:
        query = f"au:\"{author['name']}\""
        search = arxiv.Search(query=query, max_results=10, sort_by=arxiv.SortCriterion.SubmittedDate)
        try:
            for paper in client.results(search):
                paper_id = paper.entry_id.split('/abs/')[-1]
                author_papers[paper_id] = paper
        except Exception as e:
            print(f"🔴 Error fetching for author {author['name']}: {e}")

    print(f"✅ Found {len(author_papers)} papers from followed authors.")
    return list(author_papers.values())

def get_papers_from_dmrg_site():
    """
    Scrapes the DMRG site for arXiv paper IDs from the current and previous month.
    """
    print(f" scraping {DMRG_URL} for recent paper IDs...")

    # Calculate current and previous month prefixes (e.g., '2507.' and '2506.')
    today = datetime.now()
    current_month_prefix = today.strftime('%y%m.')

    prev_month_year = today.year
    prev_month = today.month - 1
    if prev_month == 0:
        prev_month = 12
        prev_month_year -= 1
    previous_month_prefix = f"{str(prev_month_year)[-2:]}{prev_month:02d}."

    print(f"  -> Filtering for prefixes: {current_month_prefix}, {previous_month_prefix}")

    try:
        response = requests.get(DMRG_URL, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        paper_ids = set()
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            if 'arxiv.org/abs/' in href:
                paper_id = href.split('/abs/')[-1]
                # Check if the paper ID matches the desired date prefixes
                if paper_id.startswith(current_month_prefix) or paper_id.startswith(previous_month_prefix):
                    paper_ids.add(paper_id)

        print(f"✅ Found {len(paper_ids)} unique paper IDs from DMRG site for the last two months.")
        return paper_ids
    except requests.exceptions.RequestException as e:
        print(f"🔴 Failed to scrape DMRG site: {e}")
        return set()


def get_llm_analysis(new_paper, liked_papers, api_key):
    """Sends a paper's details to the Gemini LLM for advanced scoring and reasoning."""
    liked_papers_details = "\n---\n".join([f"Title: {p['title']}" for p in liked_papers])
    prompt = f"""
    You are a helpful research assistant. A user has previously liked papers on these topics:
    ---
    {liked_papers_details}
    ---
    Now, consider this new paper:
    Title: {new_paper['title']}
    Abstract: {new_paper['summary']}

    Please provide a relevance score as a single number between 0.0 (not relevant) and 1.0 (highly relevant).
    Also provide a concise, one-sentence justification for your score.
    Finally, provide an array of up to 3 new, insightful keywords from this paper's abstract.

    Return the result as a single JSON object with the keys "score", "reason", and "suggested_keywords".
    """
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"responseMimeType": "application/json"}
    }
    headers = {"Content-Type": "application/json"}
    params = {"key": api_key}

    try:
        response = requests.post(GEMINI_API_URL, params=params, headers=headers, json=payload, timeout=90)
        response.raise_for_status()
        result = response.json()
        content = result['candidates'][0]['content']['parts'][0]['text']
        return json.loads(content)
    except Exception as e:
        print(f"🔴 LLM API call failed: {e}")
        return None

def update_archive():
    """Fetches, analyzes, and archives papers from all sources."""
    gemini_api_key = os.environ.get('GEMINI_API_KEY')
    if not gemini_api_key:
        print("🔴 GEMINI_API_KEY not set."); return []

    archive = json.load(open(ARCHIVE_PATH)) if os.path.exists(ARCHIVE_PATH) else {}
    taste_profile = json.load(open(TASTE_PROFILE_PATH)) if os.path.exists(TASTE_PROFILE_PATH) else []
    taste_profile_exists = bool(taste_profile)

    try:
        with open('keywords.json', 'r') as f:
            KEYWORDS = json.load(f)
    except FileNotFoundError:
        print("🔴 keywords.json not found. Assuming empty list."); KEYWORDS = []

    print("🧠 Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    liked_vectors = torch.tensor([item['vector'] for item in taste_profile]) if taste_profile_exists else None
    print("✅ Model loaded.")

    retry_ids = {
        pid for pid, p_data in archive.items()
        if (p_data.get('reasoning') == LLM_FAILURE_REASON or p_data.get('reasoning') is None) or \
           (taste_profile_exists and not p_data.get('vector_matches'))
    }
    if retry_ids:
        print(f"🔍 Found {len(retry_ids)} papers in archive to re-evaluate.")

    # Get papers from all sources
    category_papers = get_recent_papers()
    author_papers = get_papers_from_followed_authors()
    dmrg_paper_ids = get_papers_from_dmrg_site()

    # Create a unified map of all potential papers
    all_candidate_papers = {p.entry_id.split('/abs/')[-1]: p for p in category_papers}
    all_candidate_papers.update({p.entry_id.split('/abs/')[-1]: p for p in author_papers})

    # Identify new papers to process
    new_paper_ids = {
        pid: p for pid, p in all_candidate_papers.items()
        if pid not in archive and (
            any(kw.lower() in (p.title + p.summary).lower() for kw in KEYWORDS) or
            pid in {ap.entry_id.split('/abs/')[-1] for ap in author_papers} or
            pid in dmrg_paper_ids
        )
    }
    # Add any papers from the DMRG site that might not be in the recent feed
    new_from_dmrg = dmrg_paper_ids - set(all_candidate_papers.keys())

    if new_paper_ids:
        print(f"🔍 Found {len(new_paper_ids)} new papers to process from feeds.")

    # Combine all IDs that need processing
    ids_to_process = retry_ids.union(new_paper_ids.keys()).union(new_from_dmrg)

    if not ids_to_process:
        print("✅ No new or failed papers to process. Exiting."); return []

    print(f"📡 Fetching fresh data for {len(ids_to_process)} papers from arXiv...")
    client = arxiv.Client()
    papers_to_analyze = list(client.results(arxiv.Search(id_list=list(ids_to_process))))
    print(f"✅ Received fresh data for {len(papers_to_analyze)} papers.")

    delay_seconds = 0
    if len(papers_to_analyze) > LLM_RPM_LIMIT:
        delay_seconds = 60.0 / (LLM_RPM_LIMIT - 1)
        print(f"⚠️ Rate limiting LLM calls with a {delay_seconds:.2f}s delay.")

    newly_added_ids = []

    for i, paper in enumerate(papers_to_analyze):
        paper_id = paper.entry_id.split('/abs/')[-1]
        print(f"  -> ({i+1}/{len(papers_to_analyze)}) Analyzing '{paper.title[:50]}...'")

        existing_data = archive.get(paper_id)
        llm_result = None

        if existing_data and existing_data.get('reasoning') and existing_data.get('reasoning') != LLM_FAILURE_REASON:
            print(f"  -> LLM data exists. Skipping API call.")
            llm_result = {
                "score": existing_data.get('score', 0.0),
                "reason": existing_data.get('reasoning'),
                "suggested_keywords": existing_data.get('suggested_keywords', [])
            }
        else:
            llm_result = get_llm_analysis({'title': paper.title, 'summary': paper.summary}, taste_profile[:5], gemini_api_key)

        is_new_paper = paper_id not in archive

        if llm_result:
            vector_matches = []
            if liked_vectors is not None:
                new_paper_embedding = model.encode(f"Title: {paper.title}\nAbstract: {paper.summary}", convert_to_tensor=True)
                cosine_scores = util.cos_sim(new_paper_embedding, liked_vectors)[0]
                top_results = torch.topk(cosine_scores, k=min(3, len(taste_profile)))
                for score, idx in zip(top_results[0], top_results[1]):
                    vector_matches.append({
                        'score': score.item(), 'title': taste_profile[idx]['title'],
                        'authors': taste_profile[idx].get('authors', []), 'url': taste_profile[idx].get('url', '#')
                    })

            # Determine the sources for this paper
            paper_sources = existing_data.get('sources', []) if existing_data else []
            if paper_id in dmrg_paper_ids and DMRG_SOURCE_TAG not in paper_sources:
                paper_sources.append(DMRG_SOURCE_TAG)

            archive[paper_id] = {
                'id': paper_id, 'title': paper.title,
                'authors': [{'name': a.name, 'affiliation': getattr(a, 'affiliation', None)} for a in paper.authors if a],
                'summary': paper.summary.replace('\n', ' '), 'published_date': paper.published.strftime('%Y-%m-%d'),
                'categories': filter_arxiv_categories(paper.categories), 'doi': paper.doi,
                'score': llm_result.get('score', 0.0), 'reasoning': llm_result.get('reason'),
                'matching_keywords': [kw for kw in KEYWORDS if kw.lower() in (paper.title + paper.summary).lower()],
                'suggested_keywords': llm_result.get('suggested_keywords', []),
                'vector_matches': vector_matches,
                'sources': paper_sources
            }
            if is_new_paper:
                newly_added_ids.append(paper_id)
        else:
            if paper_id in archive:
                archive[paper_id]['reasoning'] = LLM_FAILURE_REASON
            else:
                archive[paper_id] = {
                    'id': paper_id, 'title': paper.title,
                    'authors': [{'name': a.name, 'affiliation': getattr(a, 'affiliation', None)} for a in paper.authors if a],
                    'summary': paper.summary.replace('\n', ' '), 'published_date': paper.published.strftime('%Y-%m-%d'),
                    'reasoning': LLM_FAILURE_REASON, 'vector_matches': [], 'score': 0, 'sources': []
                }
            print(f"  -> Marking paper {paper_id} for retry due to LLM analysis failure.")

        if delay_seconds > 0 and i < len(papers_to_analyze) - 1:
            time.sleep(delay_seconds)

    print(f"💾 Saving archive with {len(archive)} total papers...")
    with open(ARCHIVE_PATH, 'w') as f: json.dump(archive, f, indent=2)

    return newly_added_ids

def generate_site(new_paper_ids):
    """Generates the static site from the archive.json file."""
    if not os.path.exists(ARCHIVE_PATH): print("🔴 Archive file not found."); return

    with open(ARCHIVE_PATH, 'r') as f: archive = json.load(f)
    liked_paper_ids = {item['id'] for item in (json.load(open(TASTE_PROFILE_PATH)) if os.path.exists(TASTE_PROFILE_PATH) else [])}

    papers_by_month = defaultdict(list)
    for paper in archive.values():
        date_key = (paper.get('published_date') or '0000-00-00')[:7]
        papers_by_month[date_key].append(paper)

    if not papers_by_month: print("No papers in archive to generate."); return

    valid_months = [m for m in papers_by_month.keys() if re.match(r'^\d{4}-\d{2}$', m)]
    sorted_months = sorted(valid_months, reverse=True)

    env = Environment(loader=FileSystemLoader('templates'))
    env.filters['format_byl_authors'] = lambda authors: f"{authors[0]['name']}, {authors[-1]['name']} et al." if len(authors) > 2 else (' and '.join([a['name'] for a in authors]) if authors else "Unknown")
    template = env.get_template('index.html')
    github_repo = os.environ.get('GITHUB_REPOSITORY', 'yilu/arxiv-curator')

    generation_date_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')

    output_dir = 'dist'
    if os.path.exists(output_dir): shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print("📄 Generating monthly archive pages...")
    for month in sorted_months:
        papers_of_month = sorted(
            papers_by_month[month],
            key=lambda p: (p.get('score', 0), sum(match['score'] for match in p.get('vector_matches', []))),
            reverse=True
        )
        total_in_month = len(papers_of_month)
        papers_to_render = papers_of_month[:RECOMMENDATION_LIMIT] if RECOMMENDATION_LIMIT > 0 else papers_of_month

        unique_categories_in_month = sorted(list(set(cat for p in papers_to_render for cat in p.get('categories', []))))
        unique_keywords_in_month = sorted(list(set(kw for p in papers_to_render for kw in p.get('matching_keywords', []))))

        html_content = template.render(
            papers=papers_to_render, current_month=month, all_months=sorted_months,
            github_repo=github_repo, new_paper_ids=new_paper_ids,
            liked_paper_ids=liked_paper_ids, generation_date=generation_date_str,
            num_added=len(new_paper_ids), num_not_shown=max(0, total_in_month - len(papers_to_render)),
            total_in_month=total_in_month,
            filter_categories=unique_categories_in_month,
            filter_keywords=unique_keywords_in_month,
            dmrg_source_tag=DMRG_SOURCE_TAG
        )
        with open(f"{output_dir}/{month}.html", 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"  -> Created {output_dir}/{month}.html")

    if sorted_months:
        shutil.copyfile(f"{output_dir}/{sorted_months[0]}.html", f"{output_dir}/index.html")

    print("✅ Site generation complete.")

if __name__ == "__main__":
    newly_added_ids = update_archive()
    generate_site(newly_added_ids)