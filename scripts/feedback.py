# scripts/feedback.py

import os
import json
import arxiv
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL, TASTE_PROFILE_PATH

def process_feedback():
    """
    Processes feedback from a new GitHub issue, saving the source and URL.
    """
    issue_title = os.environ.get('ISSUE_TITLE')
    if not issue_title or not issue_title.startswith('Like: '):
        print("🔴 This action is only for issues starting with 'Like: '. Exiting.")
        return

    paper_id = issue_title.replace('Like: ', '').strip()
    print(f"👍 Processing like for paper ID: {paper_id}")

    if os.path.exists(TASTE_PROFILE_PATH):
        with open(TASTE_PROFILE_PATH, 'r') as f:
            taste_profile = json.load(f)
    else:
        taste_profile = []

    if any(item['id'] == paper_id for item in taste_profile):
        print(f"✅ Paper {paper_id} is already in the taste profile.")
        return

    try:
        print(f"🔍 Fetching paper '{paper_id}' from arXiv...")
        search = arxiv.Search(id_list=[paper_id])
        paper = next(arxiv.Client().results(search))
    except Exception as e:
        print(f"🔴 Error fetching paper from arXiv: {e}")
        return

    title = paper.title
    abstract = paper.summary.replace('\n', ' ')
    authors = [author.name for author in paper.authors]
    text_to_embed = f"Title: {title}\nAbstract: {abstract}"

    print(f"🧠 Loading AI model '{EMBEDDING_MODEL}'...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print(f"🧬 Generating embedding for '{title[:50]}...'")
    embedding = model.encode(text_to_embed, convert_to_tensor=False).tolist()

    taste_profile.append({
        'id': paper_id,
        'title': title,
        'authors': authors,
        'vector': embedding,
        'source': 'arxiv', # Add source
        'url': paper.entry_id # Add the correct URL
    })

    print(f"💾 Saving updated taste profile with {len(taste_profile)} vectors...")
    with open(TASTE_PROFILE_PATH, 'w') as f:
        json.dump(taste_profile, f, indent=2)

    print(f"✅ Successfully added {paper_id} to taste profile.")

if __name__ == "__main__":
    process_feedback()