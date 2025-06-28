# scripts/feedback_manager.py

import os
import sys
import json
import arxiv
import requests
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL, TASTE_PROFILE_PATH

# --- GitHub API Configuration ---
GITHUB_TOKEN = os.environ.get('GITHUB_TOKEN')
REPO = os.environ.get('GITHUB_REPOSITORY')
API_URL = f"https://api.github.com/repos/{REPO}/issues"
HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}

def get_open_feedback_issues():
    """Fetches all open 'Like' and 'Unlike' issues from the repo."""
    print("🔍 Fetching open feedback issues from GitHub...")
    params = {"state": "open", "labels": "feedback", "per_page": 100}
    try:
        response = requests.get(API_URL, headers=HEADERS, params=params)
        response.raise_for_status()
        issues = response.json()
        print(f"✅ Found {len(issues)} open feedback issues.")
        return issues
    except requests.exceptions.RequestException as e:
        print(f"🔴 Failed to fetch issues: {e}"); sys.exit(1)

def process_issues(issues):
    """Processes a list of issues, determining the final action for each paper."""
    actions = {}
    issues_to_close = []

    for issue in issues:
        title = issue.get('title', '')
        issue_number = issue.get('number')
        paper_id = None
        action = None

        if title.startswith('Like: '):
            paper_id = title.replace('Like: ', '').strip()
            action = 'like'
        elif title.startswith('Unlike: '):
            paper_id = title.replace('Unlike: ', '').strip()
            action = 'unlike'

        if paper_id and action:
            actions[paper_id] = action
            issues_to_close.append(issue_number)

    print(f"Found {len(actions)} unique paper actions to process.")
    # --- NEW: Output the issue numbers for the workflow ---
    print(f"::set-output name=issue_numbers::{json.dumps(issues_to_close)}")
    return actions

def update_taste_profile(actions):
    """Updates the liked_vectors.json file based on the processed actions."""
    if not os.path.exists(TASTE_PROFILE_PATH): taste_profile = []
    else:
        with open(TASTE_PROFILE_PATH, 'r') as f: taste_profile = json.load(f)

    ids_to_remove = {pid for pid, act in actions.items() if act == 'unlike'}
    if ids_to_remove:
        taste_profile = [p for p in taste_profile if p['id'] not in ids_to_remove]
        print(f"👎 Removed {len(ids_to_remove)} papers from taste profile.")

    ids_to_add = {pid for pid, act in actions.items() if act == 'like'}
    existing_ids = {p['id'] for p in taste_profile}
    ids_to_fetch = [pid for pid in ids_to_add if pid not in existing_ids]

    if ids_to_fetch:
        print(f"👍 Adding {len(ids_to_fetch)} new papers to taste profile...")
        model = SentenceTransformer(EMBEDDING_MODEL)
        client = arxiv.Client()
        search = arxiv.Search(id_list=ids_to_fetch)
        for paper in client.results(search):
            text_to_embed = f"Title: {paper.title}\nAbstract: {paper.summary.replace(chr(10), ' ')}"
            embedding = model.encode(text_to_embed).tolist()
            taste_profile.append({
                'id': paper.entry_id.split('/abs/')[-1], 'title': paper.title,
                'authors': [a.name for a in paper.authors], 'vector': embedding,
                'source': 'arxiv', 'url': paper.entry_id
            })
            print(f"  -> Added '{paper.title[:50]}...'")

    print(f"💾 Saving taste profile with {len(taste_profile)} total papers...")
    with open(TASTE_PROFILE_PATH, 'w') as f: json.dump(taste_profile, f, indent=2)

if __name__ == "__main__":
    if not all([GITHUB_TOKEN, REPO]):
        print("🔴 GITHUB_TOKEN and GITHUB_REPOSITORY must be set."); sys.exit(1)

    open_issues = get_open_feedback_issues()
    if not open_issues:
        print("✅ No open feedback issues to process. Exiting."); sys.exit(0)

    final_actions = process_issues(open_issues)
    update_taste_profile(final_actions)