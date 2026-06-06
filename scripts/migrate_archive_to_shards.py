# scripts/migrate_archive_to_shards.py

import os
import json
from archive_store import (
    ARCHIVE_DIR, LEGACY_ARCHIVE_PATH, save_archive, load_archive, month_key
)


def migrate_to_monthly_shards():
    """One-time migration: split the single archive.json into monthly shards.

    Reads the legacy ``archive.json`` and writes one ``archive/YYYY-MM.json`` per
    month via :func:`archive_store.save_archive`. The original ``archive.json`` is
    left in place as a backup; once a sharded run is confirmed good it can be
    deleted. Re-running is harmless (idempotent).
    """
    if not os.path.exists(LEGACY_ARCHIVE_PATH):
        print(f"🔴 {LEGACY_ARCHIVE_PATH} not found — nothing to migrate.")
        return

    print(f"🚀 Reading {LEGACY_ARCHIVE_PATH}...")
    with open(LEGACY_ARCHIVE_PATH, 'r', encoding='utf-8') as f:
        archive = json.load(f)

    if not archive:
        print("⚠️ Archive is empty — nothing to migrate.")
        return

    months = {month_key(p) for p in archive.values()}
    print(f"📦 {len(archive)} papers spanning {len(months)} month(s). Writing shards to {ARCHIVE_DIR}/...")
    save_archive(archive)

    # Verify the round-trip: every paper must reload identically from the shards.
    reloaded = load_archive()
    if reloaded == archive:
        print(f"✅ Verified: {len(reloaded)} papers reload identically from {ARCHIVE_DIR}/.")
        print(f"ℹ️  {LEGACY_ARCHIVE_PATH} kept as a backup; delete it once a sharded run looks good.")
    else:
        only_legacy = set(archive) - set(reloaded)
        only_shard = set(reloaded) - set(archive)
        print("🔴 Round-trip mismatch! Shards do NOT match the original — NOT safe to switch.")
        print(f"   {len(only_legacy)} id(s) missing from shards, {len(only_shard)} unexpected id(s).")


if __name__ == "__main__":
    migrate_to_monthly_shards()
