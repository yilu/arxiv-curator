# scripts/bootstrap.py

import os
from scholarly import scholarly
from sentence_transformers import SentenceTransformer
import json
from config import GOOGLE_SCHOLAR_USER_ID, EMBEDDING_MODEL, TASTE_PROFILE_PATH

def create_initial_profile():
    """
    Fetches publications from a Google Scholar profile, saving their source
    and correct URL along with their vector embeddings.
    """
    if not GOOGLE_SCHOLAR_USER_ID or GOOGLE_SCHOLAR_USER_ID == 'YOUR_ID_HERE':
        print("🔴 Error: Please set your GOOGLE_SCHOLAR_USER_ID in scripts/config.py")
        return

    print(f"🔍 Fetching publications for Google Scholar ID: {GOOGLE_SCHOLAR_USER_ID}")

    try:
        author = scholarly.search_author_id(GOOGLE_SCHOLAR_USER_ID)
        scholarly.fill(author, sections=['publications'])
    except Exception as e:
        print(f"🔴 Error fetching Google Scholar data: {e}")
        return

    publications = author.get('publications', [])
    if not publications:
        print("⚠️ Warning: No publications found for this user.")
        return

    print(f"✅ Found {len(publications)} publications. Preparing to generate embeddings.")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print("✅ AI model loaded.")

    taste_profile = []
    for i, pub in enumerate(publications):
        try:
            scholarly.fill(pub)
            bib = pub.get('bib', {})
            title = bib.get('title', '')
            abstract = bib.get('abstract', '')

            authors_str = bib.get('author', 'Unknown Authors')
            authors = [a.strip() for a in authors_str.split(' and ')]

            if not title or not abstract:
                continue

            text_to_embed = f"Title: {title}\nAbstract: {abstract}"
            print(f"🧬 Generating embedding for paper {i+1}/{len(publications)}: '{title[:50]}...'")
            embedding = model.encode(text_to_embed, convert_to_tensor=False).tolist()

            taste_profile.append({
                'id': pub.get('author_pub_id', title),
                'title': title,
                'authors': authors,
                'vector': embedding,
                'source': 'google_scholar',  # Add source
                'url': pub.get('pub_url', '#')   # Add the correct URL
            })
        except Exception as e:
            print(f"🔴 Error processing publication {i+1}: {e}")

    print(f"💾 Saving {len(taste_profile)} vectors to '{TASTE_PROFILE_PATH}'...")
    with open(TASTE_PROFILE_PATH, 'w') as f:
        json.dump(taste_profile, f, indent=2)

    print("✅ Bootstrap complete! Your initial taste profile has been created.")

if __name__ == "__main__":
    create_initial_profile()