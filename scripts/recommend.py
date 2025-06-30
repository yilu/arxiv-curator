# scripts/recommend.py

import os
import json
import arxiv
import shutil
import time
import requests
import numpy as np
import re
from datetime import datetime
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from jinja2 import Environment, FileSystemLoader
from config import (
    ARXIV_CATEGORIES, KEYWORDS, EMBEDDING_MODEL, TASTE_PROFILE_PATH,
    RECOMMENDATION_LIMIT, LLM_RPM_LIMIT
)

ARCHIVE_PATH = 'archive.json'
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
LLM_FAILURE_REASON = "Could not be analyzed by LLM."

def filter_arxiv_categories(categories):
    """
    Filters a list of strings to only include valid arXiv category formats.
    Example valid format: 'cs.LG', 'cond-mat.str-el'
    Example invalid format: 'I.2.5; I.2.11', '68T07'
    """
    if not categories:
        return []
    # This regex matches the standard 'archive.subject_class' format.
    arxiv_category_pattern = re.compile(r'^[a-z\-]+\.[A-Za-z\-]+$')
    return [cat for cat in categories if arxiv_category_pattern.match(cat)]

def get_recent_papers():
    """
    Fetches the most recent papers from specified arXiv categories.
    """
    print(f"🔍 Fetching recent papers from categories: {', '.join(ARXIV_CATEGORIES)}")
    all_papers = {}
    client = arxiv.Client()
    for category in ARXIV_CATEGORIES:
        print(f"Querying category '{category}'...")
        search = arxiv.Search(query=f"cat:{category}", max_results=150, sort_by=arxiv.SortCriterion.SubmittedDate)
        try:
            results = client.results(search)
            for paper in results:
                paper_id = paper.entry_id.split('/abs/')[-1]
                if paper_id not in all_papers:
                    all_papers[paper_id] = paper
        except Exception as e:
            print(f"🔴 Error fetching from {category}: {e}")
    print(f"✅ Found a total of {len(all_papers)} unique recent papers.")
    return list(all_papers.values())

def get_llm_analysis(new_paper, liked_papers, api_key):
    """
    Sends a paper's details to the Gemini LLM for advanced scoring and reasoning.
    """
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
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "OBJECT", "properties": {
                    "score": {"type": "NUMBER"}, "reason": {"type": "STRING"},
                    "suggested_keywords": {"type": "ARRAY", "items": {"type": "STRING"}}
                }, "required": ["score", "reason", "suggested_keywords"]
            }
        }
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
    """
    Fetches new papers, identifies papers needing LLM analysis (new and previously
    failed), scores them all with an LLM, and adds them to the archive.
    """
    gemini_api_key = os.environ.get('GEMINI_API_KEY')
    if not gemini_api_key:
        print("🔴 GEMINI_API_KEY not set."); return []

    archive = json.load(open(ARCHIVE_PATH)) if os.path.exists(ARCHIVE_PATH) else {}
    taste_profile = json.load(open(TASTE_PROFILE_PATH)) if os.path.exists(TASTE_PROFILE_PATH) else []
    if not taste_profile:
        print("🔴 Taste profile is empty."); return []

    # 1. Find papers already in the archive that failed or were never analyzed by the LLM.
    papers_to_retry = {
        pid: p_data for pid, p_data in archive.items()
        if p_data.get('reasoning') == LLM_FAILURE_REASON or p_data.get('reasoning') is None
    }
    if papers_to_retry:
        print(f"🔍 Found {len(papers_to_retry)} existing papers to evaluate/re-evaluate.")

    # 2. Find new papers that match keywords.
    recent_papers = get_recent_papers()
    unseen_papers = [p for p in recent_papers if p.entry_id.split('/abs/')[-1] not in archive]

    keyword_filtered_papers = [
        p for p in unseen_papers
        if any(kw.lower() in (p.title + p.summary).lower() for kw in KEYWORDS)
    ]

    # 3. Combine them into a final list for the LLM.
    papers_to_analyze_map = {p.entry_id.split('/abs/')[-1]: p for p in keyword_filtered_papers}

    for pid, p_data in papers_to_retry.items():
        papers_to_analyze_map[pid] = arxiv.Result(
            entry_id=f"http://arxiv.org/abs/{pid}",
            title=p_data['title'],
            summary=p_data['summary']
        )

    papers_to_analyze = list(papers_to_analyze_map.values())

    if not papers_to_analyze:
        print("✅ No new or failed papers to analyze with LLM."); return []

    print(f"🤖 Total papers to analyze with LLM: {len(papers_to_analyze)}")

    # 4. Smart rate-limiting logic.
    delay_seconds = 0
    if len(papers_to_analyze) > LLM_RPM_LIMIT:
        delay_seconds = 60.0 / (LLM_RPM_LIMIT - 1)
        print(f"⚠️ More than {LLM_RPM_LIMIT} papers to process. Adding a {delay_seconds:.2f}s delay between API calls.")

    liked_info = {item['id']: item for item in taste_profile}
    newly_added_ids = []

    for i, paper in enumerate(papers_to_analyze):
        paper_id = paper.entry_id.split('/abs/')[-1]
        print(f"  -> ({i+1}/{len(papers_to_analyze)}) Analyzing '{paper.title[:50]}...'")

        llm_result = get_llm_analysis(
            {'title': paper.title, 'summary': paper.summary},
            list(liked_info.values())[:5],
            gemini_api_key
        )

        is_new_paper = paper_id not in archive

        if llm_result:
            # Get raw categories and filter them
            raw_categories = getattr(paper, 'categories', archive.get(paper_id, {}).get('categories', []))
            filtered_categories = filter_arxiv_categories(raw_categories)

            archive[paper_id] = {
                'id': paper_id, 'title': paper.title,
                'authors': [getattr(a, 'name', a) for a in paper.authors],
                'summary': paper.summary.replace('\n', ' '),
                'published_date': paper.published.strftime('%Y-%m-%d') if hasattr(paper, 'published') else archive.get(paper_id, {}).get('published_date', ''),
                'categories': filtered_categories, # <--- Use the filtered list here
                'doi': getattr(paper, 'doi', archive.get(paper_id, {}).get('doi')),
                'score': llm_result.get('score', 0.0),
                'reasoning': llm_result.get('reason'),
                'matching_keywords': [kw for kw in KEYWORDS if kw.lower() in (paper.title + paper.summary).lower()],
                'suggested_keywords': llm_result.get('suggested_keywords', []),
                'vector_matches': []
            }
            if is_new_paper:
                newly_added_ids.append(paper_id)
        else:
            if paper_id in archive:
                archive[paper_id]['reasoning'] = LLM_FAILURE_REASON
            print(f"  -> Skipping/marking paper {paper_id} for retry due to LLM analysis failure.")

        if delay_seconds > 0 and i < len(papers_to_analyze) - 1:
            print(f"  -> Waiting {delay_seconds:.2f} seconds...")
            time.sleep(delay_seconds)

    print(f"💾 Saving archive with {len(archive)} total papers...")
    with open(ARCHIVE_PATH, 'w') as f: json.dump(archive, f, indent=2)

    return newly_added_ids

def generate_site(new_paper_ids):
    """
    Generates the complete multi-page static site from the archive.json file.
    """
    if not os.path.exists(ARCHIVE_PATH): print("🔴 Archive file not found."); return

    with open(ARCHIVE_PATH, 'r') as f: archive = json.load(f)
    liked_paper_ids = set()
    if os.path.exists(TASTE_PROFILE_PATH):
        with open(TASTE_PROFILE_PATH, 'r') as f:
            liked_paper_ids = {item['id'] for item in json.load(f)}

    papers_by_month = defaultdict(list)
    for paper in archive.values():
        papers_by_month[paper['published_date'][:7]].append(paper)

    if not papers_by_month: print("No papers in archive to generate."); return

    sorted_months = sorted(papers_by_month.keys(), reverse=True)

    env = Environment(loader=FileSystemLoader('templates'))
    env.filters['format_byl_authors'] = lambda authors: f"{authors[0]}, {authors[-1]} et al." if len(authors) > 1 else (f"{authors[0]} et al." if authors else "Unknown")
    template = env.get_template('index.html')
    github_repo = os.environ.get('GITHUB_REPOSITORY', 'yilu/arxiv-curator')

    generation_date_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
    num_added_this_run = len(newly_added_ids)

    output_dir = 'dist'
    if os.path.exists(output_dir): shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print("📄 Generating monthly archive pages...")
    for month in sorted_months:
        papers_of_month = sorted(papers_by_month[month], key=lambda p: p['score'], reverse=True)
        total_in_month = len(papers_of_month)

        num_not_shown = 0
        if RECOMMENDATION_LIMIT > 0 and total_in_month > RECOMMENDATION_LIMIT:
            num_not_shown = total_in_month - RECOMMENDATION_LIMIT
        papers_to_render = papers_of_month[:RECOMMENDATION_LIMIT] if RECOMMENDATION_LIMIT > 0 else papers_of_month

        unique_categories_in_month = sorted(list(set(cat for p in papers_to_render for cat in p['categories'])))
        unique_keywords_in_month = sorted(list(set(kw for p in papers_to_render for kw in p['matching_keywords'])))

        html_content = template.render(
            papers=papers_to_render, current_month=month, all_months=sorted_months,
            github_repo=github_repo, new_paper_ids=newly_added_ids,
            liked_paper_ids=liked_paper_ids, generation_date=generation_date_str,
            num_added=num_added_this_run, num_not_shown=num_not_shown,
            total_in_month=total_in_month,
            filter_categories=unique_categories_in_month,
            filter_keywords=unique_keywords_in_month
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