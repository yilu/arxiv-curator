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
KEYWORDS_PATH = 'keywords.json'

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
    keywords_to_add = set()
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
        elif title.startswith('Promote Keyword: '):
            keyword = title.replace('Promote Keyword: ', '').strip()
            if keyword:
                keywords_to_add.add(keyword.lower())
            issues_to_close.append(issue_number)
        elif title.startswith('Follow Author: '):
            author_name = title.replace('Follow Author: ', '').strip()
            if author_name:
                authors_to_follow.add(author_name)
            issues_to_close.append(issue_number)

    print(f"Found {len(paper_actions)} paper actions, {len(keywords_to_add)} keywords, and {len(authors_to_follow)} authors.")
    return paper_actions, list(keywords_to_add), list(authors_to_follow), issues_to_close

def update_keywords(new_keywords):
    """Updates the keywords.json file with new keywords."""
    if not new_keywords:
        return

    keywords = []
    if os.path.exists(KEYWORDS_PATH) and os.path.getsize(KEYWORDS_PATH) > 0:
        try:
            with open(KEYWORDS_PATH, 'r') as f:
                keywords = json.load(f)
        except json.JSONDecodeError:
            print(f"⚠️  Could not decode {KEYWORDS_PATH}, starting fresh.")
            keywords = []

    added_count = 0
    for keyword in new_keywords:
        if keyword not in keywords:
            keywords.append(keyword)
            added_count += 1

    if added_count > 0:
        print(f"🔑 Promoting {added_count} new keyword(s)...")
        keywords.sort()
        with open(KEYWORDS_PATH, 'w') as f:
            json.dump(keywords, f, indent=2)
    else:
        print("✅ No new keywords to add.")

def update_followed_authors(new_authors):
    """Updates the followed_authors.json file."""
    if not new_authors:
        return

    followed = []
    if os.path.exists(FOLLOWED_AUTHORS_PATH) and os.path.getsize(FOLLOWED_AUTHORS_PATH) > 0:
        try:
            with open(FOLLOWED_AUTHORS_PATH, 'r') as f:
                followed = json.load(f)
        except json.JSONDecodeError:
            print(f"⚠️  Could not decode {FOLLOWED_AUTHORS_PATH}, starting fresh.")
            followed = []

    followed_names = {author.get('name') for author in followed}
    added_count = 0

    for author_name in new_authors:
        if author_name not in followed_names:
            followed.append({'name': author_name})
            added_count += 1

    if added_count > 0:
        print(f"👤 Following {added_count} new author(s)...")
        followed.sort(key=lambda x: x['name'])
        with open(FOLLOWED_AUTHORS_PATH, 'w') as f:
            json.dump(followed, f, indent=2)
    else:
        print("✅ No new authors to follow.")

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
                'authors': [{'name': a.name, 'affiliation': getattr(a, 'affiliation', None)} for a in paper.authors],
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

    paper_actions, new_keywords, new_authors, issues_to_close = process_issues(open_issues)

    update_keywords(new_keywords)
    update_followed_authors(new_authors)
    update_taste_profile(paper_actions)

    close_issues(issues_to_close)