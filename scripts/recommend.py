# scripts/recommend.py

import os
import json
import arxiv
import shutil
import requests
import numpy as np
from datetime import datetime
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from jinja2 import Environment, FileSystemLoader
from config import (
    ARXIV_CATEGORIES, KEYWORDS, EMBEDDING_MODEL, TASTE_PROFILE_PATH,
    RECOMMENDATION_LIMIT, LLM_RE_RANK_LIMIT
)

ARCHIVE_PATH = 'archive.json'
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

def get_recent_papers():
    """
    Fetches the most recent papers from specified arXiv categories by querying
    each category individually and combining the unique results.
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

def find_matching_keywords(paper_text, keywords):
    """
    Checks a paper's text for any of the user-defined keywords.
    Returns a list of keywords that were found.
    """
    return [kw for kw in keywords if kw.lower() in paper_text.lower()]

def get_llm_analysis(new_paper, liked_papers, api_key):
    """
    Sends a paper's details to the Gemini LLM for advanced scoring,
    reasoning, and keyword suggestion.
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

    Please provide three things in a JSON object:
    1.  "score": A relevance score from 0.0 to 1.0.
    2.  "reason": A concise, one-sentence justification for the score.
    3.  "suggested_keywords": An array of up to 3 new, insightful keywords from this paper's abstract that are not in the user's original list.
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
        return {"score": 0.0, "reason": "Could not be analyzed by LLM.", "suggested_keywords": []}

def update_archive():
    """
    Performs the main logic of fetching new papers, scoring them, and adding
    them to the persistent archive file. Returns a list of newly added paper IDs.
    """
    gemini_api_key = os.environ.get('GEMINI_API_KEY')
    if not gemini_api_key:
        print("🔴 GEMINI_API_KEY not set."); return []

    # Load existing archive and taste profile
    archive = json.load(open(ARCHIVE_PATH)) if os.path.exists(ARCHIVE_PATH) else {}
    taste_profile = json.load(open(TASTE_PROFILE_PATH)) if os.path.exists(TASTE_PROFILE_PATH) else []
    if not taste_profile:
        print("🔴 Taste profile is empty."); return []

    # Fetch new papers and filter out those already archived
    recent_papers = get_recent_papers()
    unseen_papers = [p for p in recent_papers if p.entry_id.split('/abs/')[-1] not in archive]
    if not unseen_papers:
        print("✅ No new papers found to add to the archive."); return []

    print("🧠 Analyzing all new papers...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    new_paper_texts = [f"Title: {p.title}\nAbstract: {p.summary.replace(chr(10), ' ')}" for p in unseen_papers]
    new_paper_embeddings = model.encode(new_paper_texts)

    liked_info = {item['id']: item for item in taste_profile}
    liked_vectors = {item['id']: np.array(item['vector']) for item in taste_profile}

    newly_added_ids = []
    for i, paper in enumerate(unseen_papers):
        paper_id = paper.entry_id.split('/abs/')[-1]

        # Strict keyword filtering
        matching_keywords = find_matching_keywords(new_paper_texts[i], KEYWORDS)
        if not matching_keywords:
            continue

        # Calculate vector similarity for "because you liked" and initial score
        embedding = new_paper_embeddings[i]
        similarities = [(np.dot(embedding, v) / (np.linalg.norm(embedding) * np.linalg.norm(v)), k) for k, v in liked_vectors.items()]
        similarities.sort(key=lambda x: x[0], reverse=True)
        top_liked = similarities[:3]

        # Add the new paper to the archive with a base score
        archive[paper_id] = {
            'id': paper_id, 'title': paper.title,
            'authors': [a.name for a in paper.authors],
            'summary': paper.summary.replace('\n', ' '),
            'published_date': paper.published.strftime('%Y-%m-%d'),
            'categories': paper.categories, 'doi': paper.doi,
            'score': top_liked[0][0] if top_liked else 0.0,
            'reasoning': None,
            'matching_keywords': matching_keywords,
            'suggested_keywords': [],
            'vector_matches': [{'score': lsim, **liked_info.get(lid, {})} for lsim, lid in top_liked]
        }
        newly_added_ids.append(paper_id)

    print(f"✅ Added {len(newly_added_ids)} new papers to archive with initial vector scores.")

    # Re-rank the top candidates with the LLM for a more accurate score
    newly_added_papers = [archive[pid] for pid in newly_added_ids]
    newly_added_papers.sort(key=lambda p: p['score'], reverse=True)
    top_candidates_for_llm = newly_added_papers[:LLM_RE_RANK_LIMIT]

    print(f"🤖 Sending {len(top_candidates_for_llm)} top candidates for LLM re-ranking...")
    for paper in top_candidates_for_llm:
        print(f"  -> Re-ranking '{paper['title'][:50]}...'")
        llm_result = get_llm_analysis(paper, list(liked_info.values())[:5], gemini_api_key)

        if llm_result:
            archive[paper['id']]['score'] = llm_result.get('score', paper['score'])
            archive[paper['id']]['reasoning'] = llm_result.get('reason')
            archive[paper['id']]['suggested_keywords'] = llm_result.get('suggested_keywords', [])

    print(f"💾 Saving archive with {len(archive)} total papers...")
    with open(ARCHIVE_PATH, 'w') as f: json.dump(archive, f, indent=2)

    return newly_added_ids

def generate_site(new_paper_ids):
    """
    Generates the complete multi-page static site from the archive.json file,
    passing all necessary data to the HTML template.
    """
    if not os.path.exists(ARCHIVE_PATH):
        print("🔴 Archive file not found."); return

    with open(ARCHIVE_PATH, 'r') as f: archive = json.load(f)
    liked_paper_ids = set()
    if os.path.exists(TASTE_PROFILE_PATH):
        with open(TASTE_PROFILE_PATH, 'r') as f:
            liked_paper_ids = {item['id'] for item in json.load(f)}

    # Group all archived papers by month
    papers_by_month = defaultdict(list)
    for paper in archive.values():
        papers_by_month[paper['published_date'][:7]].append(paper)

    if not papers_by_month:
        print("No papers in archive to generate."); return

    sorted_months = sorted(papers_by_month.keys(), reverse=True)

    # Setup Jinja2 environment and custom filter
    env = Environment(loader=FileSystemLoader('templates'))
    env.filters['format_byl_authors'] = lambda authors: f"{authors[0]}, {authors[-1]} et al." if len(authors) > 1 else (f"{authors[0]} et al." if authors else "Unknown")
    template = env.get_template('index.html')
    github_repo = os.environ.get('GITHUB_REPOSITORY', 'yilu/arxiv-curator')

    generation_date_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
    num_added_this_run = len(new_paper_ids)

    # Prepare output directory
    output_dir = 'dist'
    if os.path.exists(output_dir): shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print("📄 Generating monthly archive pages...")
    for month in sorted_months:
        papers_of_month = sorted(papers_by_month[month], key=lambda p: p['score'], reverse=True)
        total_in_month = len(papers_of_month)

        # Calculate how many papers are not being shown due to the limit
        num_not_shown = 0
        if RECOMMENDATION_LIMIT > 0 and total_in_month > RECOMMENDATION_LIMIT:
            num_not_shown = total_in_month - RECOMMENDATION_LIMIT

        papers_to_render = papers_of_month[:RECOMMENDATION_LIMIT] if RECOMMENDATION_LIMIT > 0 else papers_of_month

        # Render the page for this month
        html_content = template.render(
            papers=papers_to_render,
            current_month=month,
            all_months=sorted_months,
            github_repo=github_repo,
            new_paper_ids=new_paper_ids,
            liked_paper_ids=liked_paper_ids,
            generation_date=generation_date_str,
            num_added=num_added_this_run,
            num_not_shown=num_not_shown,
            total_in_month=total_in_month
        )
        with open(f"{output_dir}/{month}.html", 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"  -> Created {output_dir}/{month}.html")

    # Create the main index.html by copying the latest month's page
    if sorted_months:
        shutil.copyfile(f"{output_dir}/{sorted_months[0]}.html", f"{output_dir}/index.html")

    print("✅ Site generation complete.")

if __name__ == "__main__":
    newly_added_ids = update_archive()
    generate_site(newly_added_ids)