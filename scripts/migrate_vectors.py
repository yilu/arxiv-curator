import json

ARCHIVE_PATH = 'archive.json'
TASTE_PROFILE_PATH = 'liked_vectors.json'

def migrate_vector_matches():
    """
    Converts the 'vector_matches' in archive.json to a more efficient format,
    storing only the paper ID and score. This is a one-time migration.
    """
    print("🚀 Starting vector_matches migration...")

    try:
        with open(ARCHIVE_PATH, 'r') as f:
            archive = json.load(f)
        with open(TASTE_PROFILE_PATH, 'r') as f:
            taste_profile = json.load(f)
    except FileNotFoundError as e:
        print(f"🔴 Error: Could not find a required file: {e}. Aborting.")
        return

    # Create a lookup map from title to ID for the liked papers
    title_to_id_map = {item['title']: item['id'] for item in taste_profile}
    updated_count = 0

    for paper_id, paper_data in archive.items():
        if 'vector_matches' in paper_data and paper_data['vector_matches']:
            # Check if the first match is in the old, detailed format
            if 'title' in paper_data['vector_matches'][0]:
                new_matches = []
                for match in paper_data['vector_matches']:
                    # Find the corresponding ID from the liked papers
                    liked_paper_id = title_to_id_map.get(match['title'])
                    if liked_paper_id:
                        new_matches.append({
                            'score': match['score'],
                            'liked_paper_id': liked_paper_id
                        })

                archive[paper_id]['vector_matches'] = new_matches
                updated_count += 1
                print(f"  -> Migrated vector_matches for paper: {paper_id}")

    if updated_count > 0:
        print(f"✅ Migrated {updated_count} entries.")
        print(f"💾 Saving updated archive to {ARCHIVE_PATH}...")
        with open(ARCHIVE_PATH, 'w') as f:
            json.dump(archive, f, indent=2)
        print("🎉 Migration complete!")
    else:
        print("✅ No entries needed migration. Format is already up to date.")

if __name__ == "__main__":
    migrate_vector_matches()