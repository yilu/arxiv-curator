# scripts/migrate_author_format.py

import json

ARCHIVE_PATH = 'archive.json'

def migrate_author_format_to_name_only():
    """
    Updates the author format in archive.json from a list of strings
    to a list of objects, each with a 'name' key. This is a one-time
    migration script to fix the display format without fetching new data.
    """
    print(f"🚀 Starting author format migration for {ARCHIVE_PATH}...")

    try:
        with open(ARCHIVE_PATH, 'r') as f:
            archive = json.load(f)
    except FileNotFoundError:
        print(f"🔴 Error: {ARCHIVE_PATH} not found. Nothing to migrate.")
        return
    except json.JSONDecodeError:
        print(f"🔴 Error: Could not decode {ARCHIVE_PATH}. Please check if it's a valid JSON file.")
        return

    if not archive:
        print("✅ Archive is empty. No migration needed.")
        return

    updated_count = 0
    for paper_id, paper_data in archive.items():
        # Check if the 'authors' field exists and is a non-empty list
        if 'authors' in paper_data and paper_data['authors']:
            # Check if the first author is a string (the old format)
            if isinstance(paper_data['authors'][0], str):
                print(f"  -> Migrating authors for paper: {paper_id}")
                # Convert list of strings to list of objects with only a name
                new_authors = [{'name': name} for name in paper_data['authors']]
                archive[paper_id]['authors'] = new_authors
                updated_count += 1

    if updated_count > 0:
        print(f"✅ Found and updated {updated_count} entries to the new author format.")
        print(f"💾 Saving updated data back to {ARCHIVE_PATH}...")
        with open(ARCHIVE_PATH, 'w') as f:
            json.dump(archive, f, indent=2)
        print("🎉 Migration complete!")
    else:
        print("✅ Archive is already in the new name-only author format. No changes needed.")

if __name__ == "__main__":
    migrate_author_format_to_name_only()
