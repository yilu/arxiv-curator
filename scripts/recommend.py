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
    GENERATED_HTML_PATH
)

def get_recent_papers():
    """Fetches papers from the last 2 days from specified arXiv categories."""
    print(f"🔍 Fetching recent papers from categories: {', '.join(ARXIV_CATEGORIES)}")

    yesterday = (datetime.now() - timedelta(days=2)).strftime('%Y%m%d')
    query = f"cat:({' OR '.join(ARXIV_CATEGORIES)}) AND submittedDate:[{yesterday} TO *]"

    search = arxiv.Search(
        query=query,
        max_results=200, # Limit the number of initial results
        sort_by=arxiv.SortCriterion.SubmittedDate
    )

    papers = list(search.results())
    print(f"✅ Found {len(papers)} recent papers.")
    return papers

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
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def generate_recommendations():
    """The main function to generate and save the recommendation HTML page."""

    # 1. Load Taste Profile
    if not os.path.exists(TASTE_PROFILE_PATH):
        print(f"🔴 Error: Taste profile '{TASTE_PROFILE_PATH}' not found.")
        print("Please run the bootstrap script first to create it from your Google Scholar profile.")
        return

    with open(TASTE_PROFILE_PATH, 'r') as f:
        taste_profile = json.load(f)

    if not taste_profile:
        print("🔴 Error: Taste profile is empty.")
        return

    print(f"✅ Loaded taste profile with {len(taste_profile)} liked papers.")

    # Create an average "taste vector"
    taste_vectors = np.array([item['vector'] for item in taste_profile])
    taste_profile_vector = np.mean(taste_vectors, axis=0)

    # 2. Fetch and Filter New Papers
    recent_papers = get_recent_papers()
    filtered_papers = filter_papers_by_keywords(recent_papers)

    if not filtered_papers:
        print("No new papers to recommend today.")
        # Still generate an empty page to show it ran
        render_page([])
        return

    # 3. Generate Embeddings for New Papers
    print(f"🧠 Loading AI model '{EMBEDDING_MODEL}'...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print(f"🧬 Generating embeddings for {len(filtered_papers)} new papers...")
    new_paper_texts = [f"Title: {p.title}\nAbstract: {p.summary.replace(chr(10), ' ')}" for p in filtered_papers]
    new_paper_embeddings = model.encode(new_paper_texts, show_progress_bar=True)

    # 4. Score and Rank Papers
    print("💯 Scoring and ranking papers...")
    scored_papers = []
    for i, paper in enumerate(filtered_papers):
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

    # Sort papers by score in descending order
    ranked_papers = sorted(scored_papers, key=lambda p: p['score'], reverse=True)

    # 5. Render HTML Page
    render_page(ranked_papers)

def render_page(papers):
    """Renders the HTML template with the list of papers."""
    print("📄 Rendering final HTML page...")

    env = Environment(loader=FileSystemLoader('templates'))
    template = env.get_template('index.html')

    github_repo = os.environ.get('GITHUB_REPOSITORY', 'yilu/arxiv-curator')

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
