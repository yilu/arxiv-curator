# scripts/recommend.py

import os
import json
import arxiv
import numpy as np
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer
from jinja2 import Environment, FileSystemLoader
from config import (
    ARXIV_CATEGORIES, KEYWORDS, EMBEDDING_MODEL, TASTE_PROFILE_PATH,
    GENERATED_HTML_PATH, SEEN_PAPERS_LIMIT
)

SEEN_PAPERS_PATH = 'seen_papers.json'

def get_recent_papers():
    """
    Fetches the most recent papers from specified arXiv categories individually
    by sorting, which is more reliable than date-range filtering.
    """
    print(f"🔍 Fetching recent papers from categories: {', '.join(ARXIV_CATEGORIES)}")
    all_papers = {}

    client = arxiv.Client()

    for category in ARXIV_CATEGORIES:
        print(f"Querying category '{category}' for most recent papers...")

        # Sort by submittedDate and take the latest results.
        search = arxiv.Search(
            query=f"cat:{category}",
            max_results=200, # Fetch the 100 most recent papers from this category
            sort_by=arxiv.SortCriterion.SubmittedDate
        )

        try:
            results = client.results(search)

            # Use a dictionary to automatically handle de-duplication
            for paper in results:
                paper_id = paper.entry_id.split('/abs/')[-1]
                if paper_id not in all_papers:
                    all_papers[paper_id] = paper
            print(f"  -> Found {len(all_papers)} unique papers so far.")
        except Exception as e:
            print(f"🔴 An error occurred while fetching from category {category}: {e}")
            continue

    final_paper_list = list(all_papers.values())
    print(f"✅ Found a total of {len(final_paper_list)} unique recent papers.")
    return final_paper_list


def filter_papers_by_keywords(papers):
    """Filters a list of papers based on keywords in title or abstract."""
    if not KEYWORDS:
        return papers

    print(f"Filtering papers with keywords: {', '.join(KEYWORDS)}")
    filtered_papers = []
    for paper in papers:
        text = (paper.title + paper.summary).lower()
        if any(keyword.lower() in text for keyword in KEYWORDS):
            filtered_papers.append(paper)

    print(f"✅ {len(filtered_papers)} papers remain after keyword filtering.")
    return filtered_papers

def calculate_similarity(v1, v2):
    """Calculates cosine similarity between two vectors."""
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return 0.0
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def generate_recommendations():
    """The main function to generate and save the recommendation HTML page."""

    # 1. Load Taste Profile and Seen Papers
    liked_paper_ids = set()
    if os.path.exists(TASTE_PROFILE_PATH):
        with open(TASTE_PROFILE_PATH, 'r') as f:
            taste_profile = json.load(f)
        liked_paper_ids = {item['id'] for item in taste_profile}
        print(f"✅ Loaded taste profile with {len(taste_profile)} liked papers.")
    else:
        taste_profile = []
        print("⚠️ Taste profile not found. It will be created on first like.")

    seen_paper_list = []
    if os.path.exists(SEEN_PAPERS_PATH):
        with open(SEEN_PAPERS_PATH, 'r') as f:
            seen_paper_list = json.load(f)
        print(f"✅ Loaded seen list with {len(seen_paper_list)} papers.")

    seen_paper_ids = set(seen_paper_list)

    # Create the taste vector for scoring
    if taste_profile:
        taste_vectors = np.array([item['vector'] for item in taste_profile])
        taste_profile_vector = np.mean(taste_vectors, axis=0)
    else:
        taste_profile_vector = None
        print("⚠️ Taste profile is empty. Recommendations will be based on keywords only.")

    # 2. Fetch and De-duplicate New Papers
    recent_papers = get_recent_papers()

    papers_to_process = []
    ignore_ids = liked_paper_ids.union(seen_paper_ids)

    for paper in recent_papers:
        paper_id = paper.entry_id.split('/abs/')[-1]
        if paper_id not in ignore_ids:
            papers_to_process.append(paper)

    print(f"✅ {len(papers_to_process)} papers remain after removing seen and liked papers.")

    # 3. Filter by Keywords
    filtered_papers = filter_papers_by_keywords(papers_to_process)

    # 4. Generate Embeddings for New Papers (handles empty list)
    new_paper_embeddings = []
    if filtered_papers:
        print(f"🧠 Loading AI model '{EMBEDDING_MODEL}'...")
        model = SentenceTransformer(EMBEDDING_MODEL)
        print(f"🧬 Generating embeddings for {len(filtered_papers)} new papers...")
        new_paper_texts = [f"Title: {p.title}\nAbstract: {p.summary.replace(chr(10), ' ')}" for p in filtered_papers]
        new_paper_embeddings = model.encode(new_paper_texts, show_progress_bar=True)

    # 5. Score and Rank Papers (handles empty list)
    scored_papers = []
    if filtered_papers:
        print("💯 Scoring and ranking papers...")
        for i, paper in enumerate(filtered_papers):
            score = 0.0
            if taste_profile_vector is not None:
                score = calculate_similarity(new_paper_embeddings[i], taste_profile_vector)

            paper_id = paper.entry_id.split('/abs/')[-1]

            scored_papers.append({
                'id': paper_id,
                'title': paper.title,
                'authors': [author.name for author in paper.authors],
                'summary': paper.summary.replace('\n', ' '),
                'published_date': paper.published.strftime('%Y-%m-%d'),
                'score': score
            })

    ranked_papers = sorted(scored_papers, key=lambda p: p['score'], reverse=True)

    # 6. Render HTML Page
    render_page(ranked_papers)

    # 7. Update and save the seen list with a cap
    newly_seen_ids = [p['id'] for p in ranked_papers]

    updated_seen_list = seen_paper_list + newly_seen_ids

    if len(updated_seen_list) > SEEN_PAPERS_LIMIT:
        print(f"⚠️ Seen list exceeds limit of {SEEN_PAPERS_LIMIT}. Trimming oldest entries.")
        updated_seen_list = updated_seen_list[-SEEN_PAPERS_LIMIT:]

    print(f"💾 Saving seen list with {len(updated_seen_list)} papers...")
    with open(SEEN_PAPERS_PATH, 'w') as f:
        json.dump(updated_seen_list, f)

def render_page(papers):
    """Renders the HTML template with the list of papers."""
    print("📄 Rendering final HTML page...")

    env = Environment(loader=FileSystemLoader('templates'))
    template = env.get_template('index.html')

    github_repo = os.environ.get('GITHUB_REPOSITORY', 'your_username/arxiv-curator')

    html_content = template.render(
        papers=papers,
        generation_date=datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC'),
        github_repo=github_repo
    )

    os.makedirs(os.path.dirname(GENERATED_HTML_PATH), exist_ok=True)
    with open(GENERATED_HTML_PATH, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"✅ Successfully generated recommendation page at '{GENERATED_HTML_PATH}'")


if __name__ == "__main__":
    generate_recommendations()
