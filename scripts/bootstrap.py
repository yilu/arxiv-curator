# scripts/bootstrap.py

import os
from scholarly import scholarly
from sentence_transformers import SentenceTransformer
import json
from config import GOOGLE_SCHOLAR_USER_ID, EMBEDDING_MODEL, TASTE_PROFILE_PATH

def create_initial_profile():
    """
    Fetches publications from a Google Scholar profile, generates their
    vector embeddings, and saves them to a file. This is a one-time
    operation to bootstrap the taste profile.
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

    print(f"🧠 Loading AI model '{EMBEDDING_MODEL}'... (This may take a moment)")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print("✅ AI model loaded.")

    taste_profile = []
    for i, pub in enumerate(publications):
        try:
            scholarly.fill(pub) # Fill in details like abstract
            title = pub.get('bib', {}).get('title', '')
            abstract = pub.get('bib', {}).get('abstract', '')

            if not title or not abstract:
                print(f"⚠️ Skipping publication {i+1} due to missing title or abstract.")
                continue

            # The text to be embedded
            text_to_embed = f"Title: {title}\nAbstract: {abstract}"

            print(f"🧬 Generating embedding for paper {i+1}/{len(publications)}: '{title[:50]}...'")
            embedding = model.encode(text_to_embed, convert_to_tensor=False).tolist()

            taste_profile.append({
                'id': pub.get('author_pub_id', title), # Use author_pub_id or title as an ID
                'title': title,
                'vector': embedding
            })
        except Exception as e:
            print(f"🔴 Error processing publication {i+1}: {e}")

    if not taste_profile:
        print("🔴 Error: Could not generate any embeddings.")
        return

    print(f"💾 Saving {len(taste_profile)} vectors to '{TASTE_PROFILE_PATH}'...")
    with open(TASTE_PROFILE_PATH, 'w') as f:
        json.dump(taste_profile, f, indent=2)

    print("✅ Bootstrap complete! Your initial taste profile has been created.")

if __name__ == "__main__":
    create_initial_profile()