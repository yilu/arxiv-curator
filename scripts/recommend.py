# scripts/recommend.py

import os
import json
import shutil
import time
import re
from datetime import datetime, timezone
from collections import defaultdict
from jinja2 import Environment, FileSystemLoader
from config import (
    ARXIV_CATEGORIES, EMBEDDING_MODEL, TASTE_PROFILE_PATH,
    RECOMMENDATION_LIMIT, LLM_RPM_LIMIT, LLM_DAILY_LIMIT, LLM_BATCH_SIZE,
    DMRG_URL, DMRG_SOURCE_TAG, DMRG_SITE_LIMIT, ARCHIVE_START_DATE,
    GEMINI_MODEL
)
from archive_store import load_archive, save_archive

# Heavy data-fetching/ML deps (arxiv, requests, bs4, torch, sentence-transformers)
# are imported lazily inside the functions that use them. This lets generate_site
# — and the "nothing new to process" early-exit — run without importing or even
# installing torch, which matters for the lightweight site-only regen workflow.

FOLLOWED_AUTHORS_PATH = 'followed_authors.json'
GEMINI_MODEL_NAME = os.environ.get("GEMINI_MODEL", GEMINI_MODEL)
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL_NAME}:generateContent"
LLM_FAILURE_REASON = "Could not be analyzed by LLM."

START_DATE = datetime.strptime(ARCHIVE_START_DATE, '%Y-%m-%d').replace(tzinfo=timezone.utc)

# --- arXiv client configuration ---
# A single, shared client with gentle defaults. Shared CI runner IPs get
# rate-limited (HTTP 429) easily, so we use a longer inter-request delay and
# more internal retries than the library defaults (3s / 3 retries).
ARXIV_PAGE_DELAY = 5.0     # seconds between paginated requests
ARXIV_NUM_RETRIES = 5      # library-level retries per page
ARXIV_ID_CHUNK_SIZE = 20   # max IDs per id_list request (large lists trigger 429s)

_ARXIV_CLIENT = None

def get_arxiv_client():
    """Lazily build and cache the shared arXiv client (defers the arxiv import)."""
    global _ARXIV_CLIENT
    if _ARXIV_CLIENT is None:
        import arxiv
        _ARXIV_CLIENT = arxiv.Client(
            page_size=100,
            delay_seconds=ARXIV_PAGE_DELAY,
            num_retries=ARXIV_NUM_RETRIES,
        )
    return _ARXIV_CLIENT

def fetch_papers_by_ids(ids, chunk_size=ARXIV_ID_CHUNK_SIZE, max_retries=4):
    """Fetch arXiv metadata for a list of (versionless) IDs.

    Requests are split into small chunks and retried with exponential backoff.
    A chunk that keeps failing (e.g. persistent HTTP 429) is skipped rather than
    crashing the whole job — those papers simply stay flagged for the next run.

    Parameters
    ----------
    ids : iterable of str
        Versionless arXiv IDs (e.g. ``2606.01234``).
    chunk_size : int
        Maximum number of IDs per ``id_list`` request.
    max_retries : int
        Attempts per chunk before giving up on that chunk.

    Returns
    -------
    list
        arXiv ``Result`` objects that were successfully retrieved.
    """
    import arxiv
    client = get_arxiv_client()
    results = []
    ids = list(ids)
    for start in range(0, len(ids), chunk_size):
        chunk = ids[start:start + chunk_size]
        delay = 10.0  # seconds; doubled after each failed attempt
        for attempt in range(max_retries):
            try:
                batch = list(client.results(arxiv.Search(id_list=chunk)))
                results.extend(batch)
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"🔴 Giving up on {len(chunk)} IDs after {max_retries} "
                          f"attempts ({e}). They remain flagged for the next run.")
                else:
                    print(f"⚠️ arXiv id_list fetch failed ({e}); retrying in "
                          f"{delay:.0f}s [attempt {attempt + 1}/{max_retries}]...")
                    time.sleep(delay)
                    delay *= 2
    return results

def _ensure_utc(dt):
    if dt is None:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)

def is_on_or_after_start_datetime(dt):
    """Returns True if the datetime is on/after the configured start date."""
    normalized = _ensure_utc(dt)
    if normalized is None:
        return False
    return normalized >= START_DATE

def is_on_or_after_start_date_str(date_str):
    if not date_str:
        return False
    try:
        parsed = datetime.strptime(date_str, '%Y-%m-%d').replace(tzinfo=timezone.utc)
    except ValueError:
        return False
    return is_on_or_after_start_datetime(parsed)

def filter_arxiv_categories(categories):
    """Filters a list of strings to only include valid arXiv category formats."""
    if not categories:
        return []
    arxiv_category_pattern = re.compile(r'^[a-z\-]+\.[A-Za-z\-]+$')
    return [cat for cat in categories if arxiv_category_pattern.match(cat)]

def get_recent_papers():
    """Fetches the most recent papers from specified arXiv categories."""
    import arxiv
    print(f"🔍 Fetching recent papers from categories: {', '.join(ARXIV_CATEGORIES)}")
    all_papers = {}
    client = get_arxiv_client()
    skipped_old = 0
    for category in ARXIV_CATEGORIES:
        print(f"Querying category '{category}'...")
        search = arxiv.Search(
            query=f"cat:{category}",
            max_results=150,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending,
        )
        try:
            for paper in client.results(search):
                if not is_on_or_after_start_datetime(_ensure_utc(getattr(paper, 'published', None))):
                    skipped_old += 1
                    continue
                paper_id = paper.entry_id.split('/abs/')[-1]
                if paper_id not in all_papers:
                    all_papers[paper_id] = paper
        except Exception as e:
            print(f"🔴 Error fetching from {category}: {e}")
    if skipped_old:
        print(f"⚠️ Skipped {skipped_old} papers published before {ARCHIVE_START_DATE} while fetching categories.")
    print(f"✅ Found a total of {len(all_papers)} unique recent papers.")
    return list(all_papers.values())

def get_papers_from_followed_authors():
    """Fetches recent papers from authors in followed_authors.json."""
    if not os.path.exists(FOLLOWED_AUTHORS_PATH) or os.path.getsize(FOLLOWED_AUTHORS_PATH) == 0:
        return []

    try:
        with open(FOLLOWED_AUTHORS_PATH, 'r') as f:
            followed_authors = json.load(f)
    except json.JSONDecodeError:
        print(f"⚠️  Could not decode {FOLLOWED_AUTHORS_PATH}, treating as empty.")
        return []

    if not followed_authors:
        return []

    import arxiv
    print(f"🔍 Fetching papers from {len(followed_authors)} followed author(s)...")
    author_papers = {}
    client = get_arxiv_client()
    skipped_old = 0
    for author in followed_authors:
        query = f"au:\"{author['name']}\""
        search = arxiv.Search(
            query=query,
            max_results=10,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending,
        )
        try:
            for paper in client.results(search):
                if not is_on_or_after_start_datetime(_ensure_utc(getattr(paper, 'published', None))):
                    skipped_old += 1
                    continue
                paper_id = paper.entry_id.split('/abs/')[-1]
                author_papers[paper_id] = paper
        except Exception as e:
            print(f"🔴 Error fetching for author {author['name']}: {e}")

    if skipped_old:
        print(f"⚠️ Skipped {skipped_old} older papers from followed authors (before {ARCHIVE_START_DATE}).")
    print(f"✅ Found {len(author_papers)} papers from followed authors.")
    return list(author_papers.values())

def get_papers_from_dmrg_site():
    """
    Scrapes the DMRG site for arXiv paper IDs from the current and previous month,
    and caps the result to the most recent ones.
    """
    import requests
    from bs4 import BeautifulSoup
    print(f" scraping {DMRG_URL} for recent paper IDs...")

    today = datetime.now()
    current_month_prefix = today.strftime('%y%m.')

    prev_month_year = today.year
    prev_month = today.month - 1
    if prev_month == 0:
        prev_month = 12
        prev_month_year -= 1
    previous_month_prefix = f"{str(prev_month_year)[-2:]}{prev_month:02d}."

    print(f"  -> Filtering for prefixes: {current_month_prefix}, {previous_month_prefix}")

    try:
        response = requests.get(DMRG_URL, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        paper_ids = []
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            if 'arxiv.org/abs/' in href:
                paper_id = href.split('/abs/')[-1]
                if paper_id.startswith(current_month_prefix) or paper_id.startswith(previous_month_prefix):
                    paper_ids.append(paper_id)

        paper_ids.sort(reverse=True)
        limited_ids = paper_ids[:DMRG_SITE_LIMIT]

        print(f"✅ Found {len(limited_ids)} most recent paper IDs from DMRG site (capped at {DMRG_SITE_LIMIT}).")
        return set(limited_ids)
    except requests.exceptions.RequestException as e:
        print(f"🔴 Failed to scrape DMRG site: {e}")
        return set()


def get_llm_batch_analysis(papers, liked_papers, api_key, retry_on_429=True):
    """
    Sends multiple papers to the Gemini LLM in one call.
    Expects papers to be a list of dicts with keys: id, title, summary.
    Returns a dict mapping paper_id -> result dict.
    """
    import requests
    liked_papers_details = "\n---\n".join([f"Title: {p['title']}" for p in liked_papers])
    paper_blocks = "\n\n".join([
        f"ID: {p['id']}\nTitle: {p['title']}\nAbstract: {p['summary']}"
        for p in papers
    ])
    prompt = f"""
You are a helpful research assistant. A user has previously liked papers on these topics:
---
{liked_papers_details}
---
Now, consider the following papers:
{paper_blocks}

For each paper, return a JSON array of objects with these keys:
- "id": the ID given above
- "score": relevance between 0.0 and 1.0
- "reason": one-sentence justification
- "suggested_keywords": up to 3 new, insightful keywords

Respond ONLY with JSON.
"""
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"responseMimeType": "application/json"}
    }
    headers = {"Content-Type": "application/json"}
    params = {"key": api_key}

    try:
        response = requests.post(GEMINI_API_URL, params=params, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        content = result['candidates'][0]['content']['parts'][0]['text']
        parsed = json.loads(content)
        output = {}
        if isinstance(parsed, list):
            for item in parsed:
                pid = item.get("id")
                if pid:
                    output[pid] = {
                        "score": item.get("score"),
                        "reason": item.get("reason"),
                        "suggested_keywords": item.get("suggested_keywords", []),
                    }
        return output
    except requests.exceptions.HTTPError as e:
        status = getattr(e.response, "status_code", None)
        if status == 429 and retry_on_429:
            print("⚠️ LLM rate limited (429). Sleeping 65s and retrying once...")
            time.sleep(65)
            return get_llm_batch_analysis(papers, liked_papers, api_key, retry_on_429=False)
        try:
            print(f"🔴 LLM API call failed: {e} | Body: {e.response.text}")
        except Exception:
            print(f"🔴 LLM API call failed: {e}")
        return {}
    except Exception as e:
        print(f"🔴 LLM API call failed: {e}")
        return {}

def update_archive():
    """Fetches, analyzes, and archives papers from all sources."""
    gemini_api_key = os.environ.get('GEMINI_API_KEY')
    if not gemini_api_key:
        print("🔴 GEMINI_API_KEY not set."); return [], None

    archive = load_archive()
    taste_profile = json.load(open(TASTE_PROFILE_PATH)) if os.path.exists(TASTE_PROFILE_PATH) else []
    taste_profile_exists = bool(taste_profile)

    try:
        with open('keywords.json', 'r') as f:
            KEYWORDS = json.load(f)
    except FileNotFoundError:
        print("🔴 keywords.json not found. Assuming empty list."); KEYWORDS = []

    dmrg_paper_ids = get_papers_from_dmrg_site()

    retry_ids = {
        pid for pid, p_data in archive.items()
        if is_on_or_after_start_date_str(p_data.get('published_date')) and (
            (p_data.get('reasoning') == LLM_FAILURE_REASON or p_data.get('reasoning') is None) or
            (taste_profile_exists and not p_data.get('vector_matches')) or
            (pid.split('v')[0] in dmrg_paper_ids and DMRG_SOURCE_TAG not in p_data.get('discovery_sources', []))
        )
    }
    if retry_ids:
        print(f"🔍 Found {len(retry_ids)} papers in archive to re-evaluate.")

    category_papers = get_recent_papers()
    author_papers = get_papers_from_followed_authors()

    all_candidate_papers = {p.entry_id.split('/abs/')[-1]: p for p in category_papers}
    all_candidate_papers.update({p.entry_id.split('/abs/')[-1]: p for p in author_papers})

    author_paper_base_ids = {p.entry_id.split('/abs/')[-1].split('v')[0] for p in author_papers}

    new_paper_ids = {
        pid for pid, p in all_candidate_papers.items()
        if pid not in archive and (
            any(kw.lower() in (p.title + p.summary).lower() for kw in KEYWORDS) or
            pid.split('v')[0] in author_paper_base_ids or
            pid.split('v')[0] in dmrg_paper_ids
        )
    }

    archive_base_ids = {pid.split('v')[0] for pid in archive.keys()}
    candidate_base_ids = {pid.split('v')[0] for pid in all_candidate_papers.keys()}
    new_from_dmrg = dmrg_paper_ids - archive_base_ids - candidate_base_ids

    if new_paper_ids:
        print(f"🔍 Found {len(new_paper_ids)} new papers to process from feeds.")

    ids_to_process = retry_ids.union(new_paper_ids).union(new_from_dmrg)

    if not ids_to_process:
        print("✅ No new or failed papers to process. Exiting."); return [], archive

    # Load the embedding model lazily — only now that we know there is work to do.
    # On quiet days this avoids the (slow) torch/SentenceTransformer import+load.
    import torch
    from sentence_transformers import SentenceTransformer, util
    os.environ.pop('TRANSFORMERS_CACHE', None)  # avoid deprecated-env-var warning
    print("🧠 Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    liked_vectors = torch.tensor([item['vector'] for item in taste_profile]) if taste_profile_exists else None
    print("✅ Model loaded.")

    versionless_ids = {pid.split('v')[0] for pid in ids_to_process}

    # Reuse Result objects we already fetched this run (category/author feeds)
    # instead of re-downloading them by ID — this roughly halves arXiv traffic
    # and is the main driver of the 429 rate-limiting that was failing the job.
    candidate_by_base = {
        pid.split('v')[0]: p for pid, p in all_candidate_papers.items()
    }
    papers_to_analyze = []
    ids_needing_fetch = []
    for vid in versionless_ids:
        cached = candidate_by_base.get(vid)
        if cached is not None:
            papers_to_analyze.append(cached)
        else:
            ids_needing_fetch.append(vid)

    print(f"📡 {len(papers_to_analyze)} papers reused from feeds; "
          f"fetching {len(ids_needing_fetch)} more from arXiv by ID...")
    if ids_needing_fetch:
        papers_to_analyze.extend(fetch_papers_by_ids(ids_needing_fetch))
    print(f"✅ Have data for {len(papers_to_analyze)} papers to analyze.")

    min_llm_interval = 60.0 / max(1, LLM_RPM_LIMIT)
    print(f"⚠️ Enforcing at least {min_llm_interval:.2f}s between LLM calls (RPM limit: {LLM_RPM_LIMIT}).")
    last_llm_call_time = None
    llm_calls_made = 0
    pending_batch = []
    llm_results = {}
    pending_entries = []
    newly_added_ids = []

    def flush_batch():
        nonlocal pending_batch, llm_calls_made, last_llm_call_time
        if not pending_batch:
            return
        if llm_calls_made >= LLM_DAILY_LIMIT:
            print(f"⚠️ LLM daily limit reached ({LLM_DAILY_LIMIT}); marking {len(pending_batch)} papers for retry.")
            for item in pending_batch:
                llm_results[item['id']] = None
            pending_batch = []
            return
        if last_llm_call_time is not None:
            elapsed = time.time() - last_llm_call_time
            if elapsed < min_llm_interval:
                sleep_time = min_llm_interval - elapsed
                print(f"   ⏳ Sleeping {sleep_time:.2f}s to respect LLM RPM limit...")
                time.sleep(sleep_time)
        batch_result = get_llm_batch_analysis(pending_batch, taste_profile[:5], gemini_api_key)
        llm_calls_made += 1
        last_llm_call_time = time.time()
        llm_results.update(batch_result)
        pending_batch = []

    for i, paper in enumerate(papers_to_analyze):
        paper_id = paper.entry_id.split('/abs/')[-1]
        base_id = paper_id.split('v')[0]
        print(f"  -> ({i+1}/{len(papers_to_analyze)}) Analyzing '{paper.title[:50]}...' ({paper_id})")

        existing_data = archive.get(paper_id)
        is_new_paper = paper_id not in archive

        discovery_sources = existing_data.get('discovery_sources', []) if existing_data else []
        if base_id in dmrg_paper_ids and DMRG_SOURCE_TAG not in discovery_sources:
            discovery_sources.append(DMRG_SOURCE_TAG)

        if existing_data and existing_data.get('reasoning') and existing_data.get('reasoning') != LLM_FAILURE_REASON:
            llm_results[paper_id] = {
                "score": existing_data.get('score', 0.0),
                "reason": existing_data.get('reasoning'),
                "suggested_keywords": existing_data.get('suggested_keywords', [])
            }
        else:
            pending_batch.append({
                "id": paper_id,
                "title": paper.title,
                "summary": paper.summary
            })
            if len(pending_batch) >= LLM_BATCH_SIZE:
                flush_batch()

        pending_entries.append({
            "paper": paper,
            "discovery_sources": discovery_sources,
            "is_new_paper": is_new_paper
        })

    flush_batch()

    # Batch-encode embeddings for every paper that needs one, in a single call.
    # SentenceTransformer is far faster encoding a list at once than looping one
    # paper at a time, which is what the per-entry loop below used to do.
    embeddings_by_id = {}
    if liked_vectors is not None:
        to_embed = [
            (e["paper"].entry_id.split('/abs/')[-1], e["paper"])
            for e in pending_entries
            if llm_results.get(e["paper"].entry_id.split('/abs/')[-1])
        ]
        if to_embed:
            texts = [f"Title: {p.title}\nAbstract: {p.summary}" for _, p in to_embed]
            print(f"🧮 Encoding {len(texts)} paper embedding(s) in one batch...")
            batch_embeddings = model.encode(texts, convert_to_tensor=True, batch_size=64)
            embeddings_by_id = {pid: emb for (pid, _), emb in zip(to_embed, batch_embeddings)}

    for entry in pending_entries:
        paper = entry["paper"]
        paper_id = paper.entry_id.split('/abs/')[-1]
        llm_result = llm_results.get(paper_id)

        if llm_result:
            vector_matches = []
            new_paper_embedding = embeddings_by_id.get(paper_id)
            if new_paper_embedding is not None:
                cosine_scores = util.cos_sim(new_paper_embedding, liked_vectors)[0]
                top_results = torch.topk(cosine_scores, k=min(3, len(taste_profile)))
                for score, idx in zip(top_results[0], top_results[1]):
                    # Store the optimized format: score and liked paper's ID
                    vector_matches.append({
                        'score': score.item(),
                        'liked_paper_id': taste_profile[idx]['id']
                    })

            archive[paper_id] = {
                'id': paper_id, 'title': paper.title,
                'authors': [{'name': a.name, 'affiliation': getattr(a, 'affiliation', None)} for a in paper.authors if a],
                'summary': paper.summary.replace('\n', ' '), 'published_date': paper.published.strftime('%Y-%m-%d'),
                'categories': filter_arxiv_categories(paper.categories), 'doi': paper.doi,
                'score': llm_result.get('score', 0.0), 'reasoning': llm_result.get('reason'),
                'matching_keywords': [kw for kw in KEYWORDS if kw.lower() in (paper.title + paper.summary).lower()],
                'suggested_keywords': llm_result.get('suggested_keywords', []),
                'vector_matches': vector_matches,
                'discovery_sources': entry["discovery_sources"]
            }
            if entry["is_new_paper"]:
                newly_added_ids.append(paper_id)
        else:
            archive[paper_id] = {
                'id': paper_id, 'title': paper.title,
                'authors': [{'name': a.name, 'affiliation': getattr(a, 'affiliation', None)} for a in paper.authors if a],
                'summary': paper.summary.replace('\n', ' '), 'published_date': paper.published.strftime('%Y-%m-%d'),
                'reasoning': LLM_FAILURE_REASON, 'vector_matches': [], 'score': 0,
                'discovery_sources': entry["discovery_sources"]
            }
            print(f"  -> Marking paper {paper_id} for retry due to LLM analysis failure.")

    print(f"💾 Saving archive with {len(archive)} total papers (monthly shards)...")
    save_archive(archive)

    return newly_added_ids, archive

def format_authors_filter(authors):
    """
    A robust Jinja2 filter to format author lists, handling both
    old (list of strings) and new (list of dicts) formats.
    """
    if not authors:
        return "Unknown"

    if isinstance(authors[0], dict):
        names = [a.get('name', '') for a in authors]
    else:
        names = authors

    if len(names) > 2:
        return f"{names[0]}, {names[-1]} et al."
    else:
        return ' and '.join(names)

def generate_site(new_paper_ids, archive=None):
    """Generates the static site from the archive.

    Parameters
    ----------
    new_paper_ids : list
        IDs newly added this run (used to badge them in the UI).
    archive : dict, optional
        The in-memory archive from ``update_archive``. When omitted (e.g. the
        standalone "regenerate site" workflow) it is loaded from the monthly
        shards on disk — avoiding a redundant multi-megabyte re-parse otherwise.
    """
    if archive is None:
        archive = load_archive()
    if not archive:
        print("🔴 Archive is empty or not found."); return

    archive_for_render = {
        pid: paper for pid, paper in archive.items()
        if is_on_or_after_start_date_str(paper.get('published_date'))
    }
    skipped_old = len(archive) - len(archive_for_render)
    if skipped_old:
        print(f"⚠️ Skipping {skipped_old} papers published before {ARCHIVE_START_DATE} while generating the site.")
    if not archive_for_render:
        archive_for_render = archive  # fallback to avoid empty site if data is malformed

    # Load taste profile and create a lookup map for rendering vector matches
    taste_profile = json.load(open(TASTE_PROFILE_PATH)) if os.path.exists(TASTE_PROFILE_PATH) else []
    liked_paper_details = {item['id']: item for item in taste_profile}
    liked_paper_ids = set(liked_paper_details.keys())

    papers_by_month = defaultdict(list)
    for paper in archive_for_render.values():
        date_key = (paper.get('published_date') or '0000-00-00')[:7]
        papers_by_month[date_key].append(paper)

    if not papers_by_month: print("No papers in archive to generate."); return

    valid_months = [m for m in papers_by_month.keys() if re.match(r'^\d{4}-\d{2}$', m)]
    sorted_months = sorted(valid_months, reverse=True)

    year_to_months = defaultdict(list)
    for month in sorted_months:
        year_to_months[month[:4]].append(month)
    year_sections = [
        {"year": year, "months": months}
        for year, months in sorted(
            ((year, sorted(months, reverse=True)) for year, months in year_to_months.items()),
            key=lambda ym: ym[0],
            reverse=True
        )
    ]
    latest_month = sorted_months[0] if sorted_months else None

    env = Environment(loader=FileSystemLoader('templates'))
    env.filters['format_byl_authors'] = format_authors_filter
    template = env.get_template('index.html')
    github_repo = os.environ.get('GITHUB_REPOSITORY', 'yilu/arxiv-curator')

    generation_date_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')

    output_dir = 'dist'
    if os.path.exists(output_dir): shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print("📄 Generating monthly archive pages...")
    for month in sorted_months:
        papers_of_month = sorted(
            papers_by_month[month],
            key=lambda p: (p.get('score', 0), sum(match['score'] for match in p.get('vector_matches', []))),
            reverse=True
        )
        total_in_month = len(papers_of_month)
        papers_to_render = papers_of_month[:RECOMMENDATION_LIMIT] if RECOMMENDATION_LIMIT > 0 else papers_of_month

        unique_categories_in_month = sorted(list(set(cat for p in papers_to_render for cat in p.get('categories', []))))
        unique_keywords_in_month = sorted(list(set(kw for p in papers_to_render for kw in p.get('matching_keywords', []))))

        html_content = template.render(
            papers=papers_to_render, current_month=month, latest_month=latest_month, year_sections=year_sections,
            github_repo=github_repo, new_paper_ids=new_paper_ids,
            liked_paper_ids=liked_paper_ids,
            liked_paper_details=liked_paper_details, # Pass the lookup map to the template
            generation_date=generation_date_str,
            num_added=len(new_paper_ids), num_not_shown=max(0, total_in_month - len(papers_to_render)),
            total_in_month=total_in_month,
            filter_categories=unique_categories_in_month,
            filter_keywords=unique_keywords_in_month,
            dmrg_source_tag=DMRG_SOURCE_TAG
        )
        with open(f"{output_dir}/{month}.html", 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"  -> Created {output_dir}/{month}.html")

    if sorted_months:
        shutil.copyfile(f"{output_dir}/{sorted_months[0]}.html", f"{output_dir}/index.html")

    print("✅ Site generation complete.")

if __name__ == "__main__":
    newly_added_ids, archive = update_archive()
    generate_site(newly_added_ids, archive)
