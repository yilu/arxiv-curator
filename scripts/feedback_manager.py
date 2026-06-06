# scripts/feedback_manager.py

import os
import sys
import json
import time
import arxiv
import requests
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL, TASTE_PROFILE_PATH

# Gentle, shared arXiv client. Shared CI runner IPs get rate-limited (HTTP 429)
# easily, so use a longer inter-request delay and more retries than the library
# defaults (3s / 3 retries).
ARXIV_CLIENT = arxiv.Client(page_size=100, delay_seconds=5.0, num_retries=5)


def fetch_papers_by_ids(ids, chunk_size=20, max_retries=4):
    """Fetch arXiv metadata for IDs in small chunks with exponential backoff.

    A chunk that keeps failing (e.g. persistent HTTP 429) is skipped rather than
    crashing the job — the corresponding likes stay open and are retried next run.
    """
    results = []
    ids = list(ids)
    for start in range(0, len(ids), chunk_size):
        chunk = ids[start:start + chunk_size]
        delay = 10.0  # seconds; doubled after each failed attempt
        for attempt in range(max_retries):
            try:
                results.extend(list(ARXIV_CLIENT.results(arxiv.Search(id_list=chunk))))
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"🔴 Giving up on {len(chunk)} IDs after {max_retries} "
                          f"attempts ({e}).")
                else:
                    print(f"⚠️ arXiv fetch failed ({e}); retrying in {delay:.0f}s "
                          f"[attempt {attempt + 1}/{max_retries}]...")
                    time.sleep(delay)
                    delay *= 2
    return results

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
    issues_to_close = []          # safe to close unconditionally (no arXiv fetch)
    like_issue_by_pid = {}        # 'Like' pid -> issue number; close only once added

    for issue in issues:
        title = issue.get('title', '')
        issue_number = issue.get('number')

        if title.startswith('Like: '):
            paper_id = title.replace('Like: ', '').strip()
            paper_actions[paper_id] = 'like'
            # Closed later only if the paper actually lands in the taste profile,
            # so a transient arXiv failure doesn't silently drop the like.
            like_issue_by_pid[paper_id] = issue_number
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
    return paper_actions, list(keywords_to_add), list(authors_to_follow), issues_to_close, like_issue_by_pid

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
    """Updates liked_vectors.json based on the processed actions.

    Returns
    -------
    set
        The 'like' paper IDs that are present in the profile afterwards (already
        present, or successfully fetched and added). Likes whose arXiv fetch
        failed are NOT included, so their issues stay open for the next run.
    """
    satisfied_likes = set()
    if not actions:
        return satisfied_likes

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

    # Match likes to profile entries by *base* id (ignoring version suffix), since
    # arXiv returns a specific version (e.g. v2) that may differ from the like id.
    existing_base = {p['id'].split('v')[0] for p in taste_profile}
    satisfied_likes |= {pid for pid in ids_to_add if pid.split('v')[0] in existing_base}

    ids_to_fetch = [pid for pid in ids_to_add if pid not in satisfied_likes]
    requested_by_base = {pid.split('v')[0]: pid for pid in ids_to_fetch}

    if not ids_to_fetch:
        print("✅ No new papers to add to taste profile.")
    else:
        print(f"👍 Adding {len(ids_to_fetch)} new papers to taste profile...")
        model = SentenceTransformer(EMBEDDING_MODEL)
        for paper in fetch_papers_by_ids(ids_to_fetch):
            text_to_embed = f"Title: {paper.title}\nAbstract: {paper.summary.replace(chr(10), ' ')}"
            embedding = model.encode(text_to_embed).tolist()
            fetched_pid = paper.entry_id.split('/abs/')[-1]
            taste_profile.append({
                'id': fetched_pid,
                'title': paper.title,
                'authors': [{'name': a.name, 'affiliation': getattr(a, 'affiliation', None)} for a in paper.authors],
                'vector': embedding,
                'source': 'arxiv',
                'url': paper.entry_id
            })
            original = requested_by_base.get(fetched_pid.split('v')[0])
            if original:
                satisfied_likes.add(original)
            print(f"  -> Added '{paper.title[:50]}...'")

    with open(TASTE_PROFILE_PATH, 'w') as f:
        json.dump(taste_profile, f, indent=2)

    return satisfied_likes

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

    paper_actions, new_keywords, new_authors, issues_to_close, like_issue_by_pid = process_issues(open_issues)

    update_keywords(new_keywords)
    update_followed_authors(new_authors)
    satisfied_likes = update_taste_profile(paper_actions)

    # Close 'Like' issues only for papers that actually made it into the profile;
    # leave the rest open so a transient arXiv failure is retried next run.
    like_issues_to_close = [num for pid, num in like_issue_by_pid.items() if pid in satisfied_likes]
    skipped = len(like_issue_by_pid) - len(like_issues_to_close)
    if skipped:
        print(f"⏭️  Leaving {skipped} 'Like' issue(s) open (paper fetch failed); will retry next run.")

    close_issues(issues_to_close + like_issues_to_close)