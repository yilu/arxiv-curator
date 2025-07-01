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
FOLLOWED_AUTHORS_PATH = 'followed_authors.json'

def get_open_feedback_issues():
    """Fetches all open 'feedback' labeled issues from the repo."""
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
    """Processes issues for paper actions, keyword promotions, and author follows."""
    paper_actions = {}
    authors_to_follow = set()
    issues_to_close = []

    for issue in issues:
        title = issue.get('title', '')
        issue_number = issue.get('number')

        if title.startswith('Like: '):
            paper_id = title.replace('Like: ', '').strip()
            paper_actions[paper_id] = 'like'
            issues_to_close.append(issue_number)
        elif title.startswith('Unlike: '):
            paper_id = title.replace('Unlike: ', '').strip()
            paper_actions[paper_id] = 'unlike'
            issues_to_close.append(issue_number)
        elif title.startswith('Follow Author: '):
            author_str = title.replace('Follow Author: ', '').strip()
            if author_str:
                authors_to_follow.add(author_str)
            issues_to_close.append(issue_number)

    print(f"Found {len(paper_actions)} paper actions and {len(authors_to_follow)} authors to follow.")
    return paper_actions, list(authors_to_follow), issues_to_close

def update_followed_authors(new_authors_str):
    """Updates the followed_authors.json file."""
    if not new_authors_str:
        return

    try:
        with open(FOLLOWED_AUTHORS_PATH, 'r+') as f:
            followed = json.load(f)
            followed_set = {f"{a['name']}|{a['affiliation']}" for a in followed}
            added_count = 0

            for author_str in new_authors_str:
                if author_str not in followed_set:
                    name, _, affiliation = author_str.partition('|')
                    if affiliation == 'None':
                        affiliation = None
                    followed.append({'name': name, 'affiliation': affiliation})
                    added_count += 1

            if added_count > 0:
                print(f"👤 Following {added_count} new author(s)...")
                followed.sort(key=lambda x: x['name'])
                f.seek(0)
                json.dump(followed, f, indent=2)
                f.truncate()
            else:
                print("✅ No new authors to follow.")
    except FileNotFoundError:
        print(f"🔴 {FOLLOWED_AUTHORS_PATH} not found. Creating it.")
        with open(FOLLOWED_AUTHORS_PATH, 'w') as f:
            new_authors_list = []
            for author_str in new_authors_str:
                name, _, affiliation = author_str.partition('|')
                if affiliation == 'None':
                    affiliation = None
                new_authors_list.append({'name': name, 'affiliation': affiliation})
            json.dump(sorted(new_authors_list, key=lambda x: x['name']), f, indent=2)

def update_taste_profile(actions):
    """Updates the liked_vectors.json file based on the processed actions."""
    if not actions:
        return

    if not os.path.exists(TASTE_PROFILE_PATH):
        taste_profile = []
    else:
        with open(TASTE_PROFILE_PATH, 'r') as f:
            taste_profile = json.load(f)

    ids_to_remove = {pid for pid, act in actions.items() if act == 'unlike'}
    if ids_to_remove:
        taste_profile = [p for p in taste_profile if p['id'] not in ids_to_remove]
        print(f"👎 Removed {len(ids_to_remove)} papers from taste profile.")

    ids_to_add = {pid for pid, act in actions.items() if act == 'like'}
    existing_ids = {p['id'] for p in taste_profile}
    ids_to_fetch = [pid for pid in ids_to_add if pid not in existing_ids]

    if not ids_to_fetch:
        print("✅ No new papers to add to taste profile.")
    else:
        print(f"👍 Adding {len(ids_to_fetch)} new papers to taste profile...")
        model = SentenceTransformer(EMBEDDING_MODEL)
        client = arxiv.Client()
        search = arxiv.Search(id_list=ids_to_fetch)
        for paper in client.results(search):
            text_to_embed = f"Title: {paper.title}\nAbstract: {paper.summary.replace(chr(10), ' ')}"
            embedding = model.encode(text_to_embed).tolist()
            taste_profile.append({
                'id': paper.entry_id.split('/abs/')[-1],
                'title': paper.title,
                'authors': [{'name': a.name, 'affiliation': a.affiliation} for a in paper.authors],
                'vector': embedding,
                'source': 'arxiv',
                'url': paper.entry_id
            })
            print(f"  -> Added '{paper.title[:50]}...'")

    with open(TASTE_PROFILE_PATH, 'w') as f:
        json.dump(taste_profile, f, indent=2)

def close_issues(issue_numbers):
    """Closes all processed GitHub issues."""
    if not issue_numbers:
        print("No issues to close.")
        return

    print(f"Closing {len(issue_numbers)} processed issues...")
    for number in issue_numbers:
        close_url = f"{API_URL}/{number}"
        try:
            requests.patch(close_url, headers=HEADERS, json={"state": "closed"})
            print(f"  -> Closed issue #{number}")
        except requests.exceptions.RequestException as e:
            print(f"🔴 Failed to close issue #{number}: {e}")

if __name__ == "__main__":
    if not all([GITHUB_TOKEN, REPO]):
        print("🔴 GITHUB_TOKEN and GITHUB_REPOSITORY must be set."); sys.exit(1)

    open_issues = get_open_feedback_issues()
    if not open_issues:
        print("✅ No open feedback issues to process. Exiting."); sys.exit(0)

    paper_actions, new_authors, issues_to_close = process_issues(open_issues)

    update_followed_authors(new_authors)
    update_taste_profile(paper_actions)

    close_issues(issues_to_close)