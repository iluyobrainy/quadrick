"""
Backup core Supabase trading tables to local JSON files.

Usage:
    python database/backup_supabase.py
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv
from supabase import create_client


CORE_TABLES = [
    "trades",
    "decisions",
    "trade_memories",
    "active_contexts",
    "strategy_learning",
    "bot_status",
    "active_positions",
    "system_logs",
]


def _fetch_all_rows(client, table: str, page_size: int = 1000) -> List[Dict]:
    rows: List[Dict] = []
    start = 0
    while True:
        end = start + page_size - 1
        response = client.table(table).select("*").range(start, end).execute()
        batch = response.data or []
        rows.extend(batch)
        if len(batch) < page_size:
            break
        start += page_size
    return rows


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    load_dotenv(project_root / ".env")

    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = (
        os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        or os.getenv("SUPABASE_KEY")
        or os.getenv("SUPABASE_ANON_KEY")
    )

    if not supabase_url or not supabase_key:
        raise RuntimeError("Missing SUPABASE_URL or Supabase key in environment")

    client = create_client(supabase_url, supabase_key)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_dir = project_root / "database" / "backups" / f"supabase_{timestamp}"
    backup_dir.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, int] = {}
    for table in CORE_TABLES:
        try:
            rows = _fetch_all_rows(client, table)
            manifest[table] = len(rows)
            with open(backup_dir / f"{table}.json", "w", encoding="utf-8") as fp:
                json.dump(rows, fp, ensure_ascii=False, indent=2, default=str)
            print(f"[ok] {table}: {len(rows)} rows")
        except Exception as exc:
            manifest[table] = -1
            print(f"[warn] {table}: backup failed ({exc})")

    with open(backup_dir / "manifest.json", "w", encoding="utf-8") as fp:
        json.dump(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "tables": manifest,
            },
            fp,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Backup complete: {backup_dir}")


if __name__ == "__main__":
    main()

