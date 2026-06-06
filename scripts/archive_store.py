# scripts/archive_store.py
"""Monthly-sharded storage for the paper archive.

The archive is kept as one JSON file per month under ``archive/`` (e.g.
``archive/2026-06.json``), instead of a single multi-megabyte ``archive.json``.
Because a daily run only adds papers to the current month, only that one small
shard changes — keeping git diffs (and history growth) tiny.

Reads and writes go through this single module so the one-time migration and the
daily writer serialize identically: unchanged shards stay byte-for-byte stable
and therefore never show up as a git change.

This module deliberately has no heavy dependencies (no torch/arxiv) so it can be
imported and exercised on its own.
"""

import os
import json
from collections import defaultdict

ARCHIVE_DIR = 'archive'          # directory of monthly shards
LEGACY_ARCHIVE_PATH = 'archive.json'  # pre-sharding single file (fallback/backup)


def month_key(paper):
    """Return the ``YYYY-MM`` shard key for a paper record.

    Mirrors the grouping used when rendering the site. Papers with a missing or
    malformed ``published_date`` fall into the ``0000-00`` shard.
    """
    return (paper.get('published_date') or '0000-00-00')[:7]


def shard_path(month):
    return os.path.join(ARCHIVE_DIR, f"{month}.json")


def load_archive():
    """Load the full archive as a single ``{paper_id: record}`` dict.

    Reads every ``archive/*.json`` shard. If the shard directory does not exist
    yet (i.e. before migration), falls back to the legacy single ``archive.json``.
    """
    archive = {}
    if os.path.isdir(ARCHIVE_DIR):
        for name in sorted(os.listdir(ARCHIVE_DIR)):
            if name.endswith('.json'):
                with open(os.path.join(ARCHIVE_DIR, name), 'r', encoding='utf-8') as f:
                    archive.update(json.load(f))
    elif os.path.exists(LEGACY_ARCHIVE_PATH):
        with open(LEGACY_ARCHIVE_PATH, 'r', encoding='utf-8') as f:
            archive = json.load(f)
    return archive


def save_archive(archive):
    """Write the archive back out as deterministic monthly shards.

    ``sort_keys=True`` makes the output a pure function of the data, so months
    that did not change serialize to identical bytes and produce no git diff.
    Shard files whose papers have all disappeared are removed.
    """
    os.makedirs(ARCHIVE_DIR, exist_ok=True)

    by_month = defaultdict(dict)
    for pid, paper in archive.items():
        by_month[month_key(paper)][pid] = paper

    for month, papers in by_month.items():
        with open(shard_path(month), 'w', encoding='utf-8') as f:
            json.dump(papers, f, indent=2, sort_keys=True, ensure_ascii=False)
            f.write('\n')

    # Drop stale shards that no longer hold any papers.
    wanted = {f"{month}.json" for month in by_month}
    for name in os.listdir(ARCHIVE_DIR):
        if name.endswith('.json') and name not in wanted:
            os.remove(os.path.join(ARCHIVE_DIR, name))
