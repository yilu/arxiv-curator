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
    """Fetches the most recent papers from specified arXiv categories."""
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
                if paper_id not in all_papers: all_papers[paper_id] = paper
        except Exception as e:
            print(f"🔴 Error fetching from {category}: {e}")

    print(f"✅ Found a total of {len(all_papers)} unique recent papers.")
    return list(all_papers.values())

def filter_papers_by_keywords(papers, keywords):
    """Performs a strict filtering of papers based on keywords."""
    if not keywords: return papers
    print(f"Filtering {len(papers)} papers with keywords...")
    filtered_papers = [p for p in papers if any(k.lower() in (p.title + p.summary).lower() for k in keywords)]
    print(f"✅ {len(filtered_papers)} papers remain after keyword filtering.")
    return filtered_papers

def calculate_similarity(v1, v2):
    """Calculates cosine similarity."""
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0: return 0.0
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def get_llm_score_and_reason(new_paper, liked_papers, api_key):
    """Gets a score and reasoning from the Gemini LLM."""

    # --- FIX for f-string SyntaxError ---
    # Construct the liked papers part of the prompt separately to avoid backslashes in the expression.
    liked_papers_details = []
    for p in liked_papers:
        title = p['title']
        abstract = p.get('summary', p.get('abstract', ''))[:500]
        liked_papers_details.append(f"Liked Paper Title: {title}\nLiked Paper Abstract: {abstract}...")

    liked_papers_prompt_part = "\n---\n".join(liked_papers_details)

    prompt = f"""
    You are a helpful research assistant. A user has previously liked the following papers:
    ---
    {liked_papers_prompt_part}
    ---
    Now, consider this new paper:
    New Paper Title: {new_paper['title']}
    New Paper Abstract: {new_paper['summary']}

    Based on the user's liked papers, please provide a relevance score and a brief justification.
    The score should be a number between 0.0 (not relevant) and 1.0 (highly relevant).
    The justification should be a single, concise sentence explaining why the new paper is or is not a good recommendation.
    """

    payload = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],"generationConfig": {"responseMimeType": "application/json","responseSchema": {"type": "OBJECT","properties": {"score": {"type": "NUMBER"},"reason": {"type": "STRING"}},"required": ["score", "reason"]}}}
    headers = {"Content-Type": "application/json"}
    params = {"key": api_key}

    try:
        response = requests.post(GEMINI_API_URL, params=params, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        content = result['candidates'][0]['content']['parts'][0]['text']
        return json.loads(content)
    except Exception as e:
        print(f"🔴 LLM API call failed: {e}")
        return {"score": 0.0, "reason": "Could not be analyzed by LLM."}

def update_archive():
    """Fetches new papers, saves all relevant ones, and re-ranks the best with an LLM."""
    gemini_api_key = os.environ.get('GEMINI_API_KEY')
    if not gemini_api_key:
        print("🔴 GEMINI_API_KEY environment variable not set. Cannot perform LLM re-ranking.")
        return []

    if os.path.exists(ARCHIVE_PATH):
        with open(ARCHIVE_PATH, 'r') as f: archive = json.load(f)
    else: archive = {}

    if os.path.exists(TASTE_PROFILE_PATH):
        with open(TASTE_PROFILE_PATH, 'r') as f: taste_profile = json.load(f)
    else: taste_profile = []

    if not taste_profile:
        print("🔴 Taste profile is empty. Cannot score papers."); return []

    # Stage 1: Initial filtering
    recent_papers = get_recent_papers()
    unseen_papers = [p for p in recent_papers if p.entry_id.split('/abs/')[-1] not in archive]
    keyword_filtered_papers = filter_papers_by_keywords(unseen_papers, KEYWORDS)

    if not keyword_filtered_papers:
        print("✅ No new papers matching keywords."); return []

    # Stage 2: Save ALL keyword-filtered papers with a base vector score
    print("🧠 Analyzing all keyword-filtered papers for initial scoring...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    new_paper_texts = [f"Title: {p.title}\nAbstract: {p.summary.replace(chr(10), ' ')}" for p in keyword_filtered_papers]
    new_paper_embeddings = model.encode(new_paper_texts)

    liked_info = {item['id']: item for item in taste_profile}
    liked_vectors = {item['id']: np.array(item['vector']) for item in taste_profile}
    taste_profile_vector = np.mean([item['vector'] for item in taste_profile], axis=0)

    all_new_papers_with_scores = []
    for i, paper in enumerate(keyword_filtered_papers):
        paper_id = paper.entry_id.split('/abs/')[-1]
        embedding = new_paper_embeddings[i]
        vector_score = calculate_similarity(embedding, taste_profile_vector)

        similarities = [(calculate_similarity(embedding, v), k) for k, v in liked_vectors.items()]
        similarities.sort(key=lambda x: x[0], reverse=True)
        top_liked = similarities[:3]

        archive[paper_id] = {
            'id': paper_id, 'title': paper.title,
            'authors': [author.name for author in paper.authors],
            'summary': paper.summary.replace('\n', ' '),
            'published_date': paper.published.strftime('%Y-%m-%d'),
            'categories': paper.categories, 'doi': paper.doi,
            'score': vector_score, # Initial score
            'reasoning': None, # Default to None
            'vector_matches': [{'score': lsim, **liked_info.get(lid, {})} for lsim, lid in top_liked]
        }
        all_new_papers_with_scores.append({'id': paper_id, 'paper': paper, 'score': vector_score})

    print(f"✅ Added {len(keyword_filtered_papers)} new papers to archive with initial vector scores.")

    # Stage 3: LLM Re-ranking for the most promising candidates
    all_new_papers_with_scores.sort(key=lambda x: x['score'], reverse=True)
    top_candidates_for_llm = all_new_papers_with_scores[:LLM_RE_RANK_LIMIT]

    print(f"🤖 Sending {len(top_candidates_for_llm)} top candidates for LLM re-ranking...")
    for candidate in top_candidates_for_llm:
        paper_id = candidate['id']
        paper = candidate['paper']

        print(f"  -> Re-ranking '{paper.title[:50]}...'")

        llm_result = get_llm_score_and_reason(
            {'title': paper.title, 'summary': paper.summary},
            list(liked_info.values())[:5],
            gemini_api_key
        )

        # Update the existing archive entry with the superior LLM score and reasoning
        if llm_result:
            archive[paper_id]['score'] = llm_result.get('score', candidate['score']) # Fallback to vector score on failure
            archive[paper_id]['reasoning'] = llm_result.get('reason', 'LLM analysis failed.')

    # 4. Save the final, updated archive
    print(f"💾 Saving archive with {len(archive)} total papers...")
    with open(ARCHIVE_PATH, 'w') as f:
        json.dump(archive, f, indent=2)

    newly_added_ids = [p['id'] for p in all_new_papers_with_scores]
    return newly_added_ids

def generate_site(new_paper_ids):
    """Generates the multi-page static site from the archive."""
    if not os.path.exists(ARCHIVE_PATH): print("🔴 Archive file not found."); return

    with open(ARCHIVE_PATH, 'r') as f: archive = json.load(f)

    papers_by_month = defaultdict(list)
    for paper in archive.values():
        month_key = datetime.strptime(paper['published_date'], '%Y-%m-%d').strftime('%Y-%m')
        papers_by_month[month_key].append(paper)

    if not papers_by_month: print("No papers in archive to generate."); return

    sorted_months = sorted(papers_by_month.keys(), reverse=True)

    env = Environment(loader=FileSystemLoader('templates'))
    def format_byl_authors(authors):
        if not authors: return "Unknown Authors"
        if len(authors) == 1: return f"{authors[0]} et al."
        return f"{authors[0]}, {authors[-1]} et al."
    env.filters['format_byl_authors'] = format_byl_authors

    template = env.get_template('index.html')
    github_repo = os.environ.get('GITHUB_REPOSITORY', 'yilu/arxiv-curator')

    output_dir = 'dist'
    if os.path.exists(output_dir): shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    print("📄 Generating monthly archive pages...")
    for month in sorted_months:
        papers_of_month = sorted(papers_by_month[month], key=lambda p: p['score'], reverse=True)
        papers_to_render = papers_of_month[:RECOMMENDATION_LIMIT] if RECOMMENDATION_LIMIT > 0 else papers_of_month

        html_content = template.render(
            papers=papers_to_render, current_month=month,
            all_months=sorted_months, github_repo=github_repo,
            new_paper_ids=newly_added_ids
        )
        with open(f"{output_dir}/{month}.html", 'w', encoding='utf-8') as f: f.write(html_content)
        print(f"  -> Created {output_dir}/{month}.html with {len(papers_to_render)} papers.")

    if sorted_months:
        shutil.copyfile(f"{output_dir}/{sorted_months[0]}.html", f"{output_dir}/index.html")

    print("✅ Site generation complete.")

if __name__ == "__main__":
    newly_added_ids = update_archive()
    generate_site(newly_added_ids)