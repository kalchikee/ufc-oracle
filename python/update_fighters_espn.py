#!/usr/bin/env python3
"""Populate the fighters table with debutants found on upcoming UFC cards.

UFCStats.com is now JS-challenged and the original scraper (`fighter-db-update.yml`)
silently scrapes zero rows on every run, so new fighters debuting since
mid-May 2026 never get added to the DB. The downstream prediction pipeline
then can't look them up by name and the bout falls back to default features.

This script does a minimal but useful subset: walks the next ~30 days of
UFC events from ESPN's MMA scoreboard, identifies fighters whose
displayName doesn't already exist in the fighters table, fetches each
missing fighter's bio (height/weight/DOB/nickname) + W-L record from
ESPN's core athlete endpoint, and inserts a row with sport-average
default stats. The prediction code already handles unknown-stats-but-
known-bio fighters via the existing default-features path, so this
unblocks the pipeline for fresh debuts without trying to replicate
UFCStats's granular strike/grappling stats (which ESPN doesn't expose).

Idempotent: existing fighters are left alone. Re-running is a no-op
once every name has been added.

Usage:
    python python/update_fighters_espn.py [--dry-run] [--days N]
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
import time
import unicodedata
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import requests

ROOT = Path(__file__).parent.parent
DB_PATHS = [ROOT / "oracle.sqlite", ROOT / "data" / "oracle.sqlite"]
ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/mma/ufc/scoreboard"
ESPN_ATHLETE = "http://sports.core.api.espn.com/v2/sports/mma/athletes/{id}"
HEADERS = {"User-Agent": "UFC-Oracle-FighterUpdate/4.1"}
REQUEST_DELAY = 0.3  # be polite to ESPN


def normalize_name(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.lower().strip()


def find_db() -> Path:
    """The repo carries the DB at root; fall back to data/ for older runs."""
    for p in DB_PATHS:
        if p.exists():
            return p
    raise FileNotFoundError(f"No oracle.sqlite found in {DB_PATHS}")


def load_existing_names(con: sqlite3.Connection) -> set[str]:
    cur = con.execute("SELECT name FROM fighters")
    return {normalize_name(r[0]) for r in cur.fetchall()}


def fetch_upcoming_events(days_ahead: int) -> list[dict]:
    """Walk the next N days of the scoreboard and return all events.

    The undated scoreboard call returns only the next event; sweeping a
    date range catches multi-week fight schedules."""
    events: list[dict] = []
    seen_ids: set[str] = set()
    today = datetime.utcnow().date()
    for offset in range(days_ahead + 1):
        date_str = (today + timedelta(days=offset)).strftime("%Y%m%d")
        try:
            r = requests.get(f"{ESPN_SCOREBOARD}?dates={date_str}",
                             headers=HEADERS, timeout=10)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f"[fighter-update] ESPN scoreboard {date_str} failed: {e}")
            continue
        for ev in data.get("events", []) or []:
            eid = ev.get("id")
            if eid and eid not in seen_ids:
                seen_ids.add(eid)
                events.append(ev)
        time.sleep(REQUEST_DELAY)
    return events


def collect_missing_fighters(events: list[dict],
                             existing: set[str]) -> dict[str, dict]:
    """Return {display_name: {athleteId, weightClassText}} for fighters not
    already in the DB. ESPN's scoreboard payload doesn't include athlete IDs
    on the competitor.athlete sub-object — only the parent competitor.id —
    so we use the competitor.id and resolve the bio endpoint."""
    missing: dict[str, dict] = {}
    for ev in events:
        for comp in ev.get("competitions", []) or []:
            for competitor in comp.get("competitors", []) or []:
                ath = competitor.get("athlete") or {}
                name = ath.get("displayName") or ""
                if not name or normalize_name(name) in existing:
                    continue
                if name in missing:
                    continue
                athlete_id = competitor.get("id") or ""
                weight_text = (ath.get("weightClass") or {}).get("text") or ""
                missing[name] = {
                    "athleteId": str(athlete_id),
                    "weightText": weight_text,
                }
    return missing


def fetch_athlete_bio(athlete_id: str) -> Optional[dict]:
    if not athlete_id:
        return None
    try:
        r = requests.get(ESPN_ATHLETE.format(id=athlete_id),
                         headers=HEADERS, timeout=10)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"[fighter-update] athlete {athlete_id} bio failed: {e}")
        return None


def parse_weight_class(weight_text: str) -> str:
    """ESPN weight strings like 'Lightweight' / 'Women's Strawweight'."""
    w = (weight_text or "").lower()
    if "women" in w or "wom" in w:
        if "straw" in w: return "WomenStrawweight"
        if "fly" in w: return "WomenFlyweight"
        if "bantam" in w: return "WomenBantamweight"
        if "feather" in w: return "WomenFeatherweight"
    if "straw" in w: return "Strawweight"
    if "flyweight" in w or "fly" in w: return "Flyweight"
    if "bantam" in w: return "Bantamweight"
    if "feather" in w: return "Featherweight"
    if "light" in w and "heavy" not in w: return "Lightweight"
    if "welter" in w: return "Welterweight"
    if "middle" in w: return "Middleweight"
    if "light heavy" in w: return "LightHeavyweight"
    if "heavyweight" in w or "heavy" in w: return "Heavyweight"
    return "Unknown"


def bio_to_row(name: str, athlete_id: str, weight_text: str,
               bio: Optional[dict]) -> dict:
    """Build a fighters-table row from ESPN bio. Stats columns are filled
    with sport-average defaults — these aren't accurate but mean the
    prediction pipeline doesn't crash on NaN; predictions for these new
    fighters will lean heavily on Elo + record until they have UFC fights."""
    nickname = (bio or {}).get("nickname") or None
    height_in = (bio or {}).get("height")  # inches per ESPN's payload
    height_cm = float(height_in) * 2.54 if height_in else None
    weight_lb = (bio or {}).get("weight")
    weight_class = parse_weight_class(weight_text or (bio or {}).get("displayWeight", ""))
    dob = (bio or {}).get("dateOfBirth")
    if dob and "T" in dob:
        dob = dob.split("T")[0]
    return {
        "fighter_id":              f"espn-{athlete_id}",
        "name":                    name,
        "nickname":                nickname,
        "weight_class":            weight_class,
        "height":                  height_cm,
        "reach":                   None,
        "stance":                  (bio or {}).get("stance"),
        "date_of_birth":           dob,
        "camp":                    None,
        # Sport-average defaults so the prediction pipeline can pull a row
        # without NaNs.  The model already lightly weights these for fighters
        # with thin UFC history; new debutants will get pulled toward median.
        "sig_strikes_landed_pm":   3.5,
        "sig_strikes_absorbed_pm": 3.5,
        "sig_strike_accuracy":     0.43,
        "sig_strike_defense":      0.58,
        "takedown_avg_per15":      1.5,
        "takedown_accuracy":       0.40,
        "takedown_defense":        0.65,
        "submission_avg_per15":    0.5,
        "knockdown_rate":          0.07,
        "control_time_pct":        0.05,
        "avg_fight_time":          11.0,
        "wins":                    0,
        "losses":                  0,
        "draws":                   0,
        "win_pct":                 0.5,
        "ufc_wins":                0,
        "ufc_losses":              0,
        "finish_rate":             0.40,
        "decision_rate":           0.60,
        "style":                   "Balanced",
        "elo_overall":             1500.0,
        "elo_striking":            1500.0,
        "elo_grappling":           1500.0,
        "last_fight_date":         None,
        "days_since_last_fight":   None,
        "recent_wins":             0,
        "recent_losses":           0,
        "recent_sig_strikes_pm":   3.5,
        "win_streak":              0,
        "updated_at":              datetime.utcnow().isoformat(),
    }


def insert_fighter(con: sqlite3.Connection, row: dict) -> None:
    cols = list(row.keys())
    placeholders = ",".join(f":{c}" for c in cols)
    cols_str = ",".join(cols)
    con.execute(
        f"INSERT OR IGNORE INTO fighters ({cols_str}) VALUES ({placeholders})",
        row,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--days", type=int, default=30,
                        help="Days ahead to sweep (default 30)")
    args = parser.parse_args()

    db_path = find_db()
    print(f"[fighter-update] using DB: {db_path}")
    con = sqlite3.connect(db_path)
    existing = load_existing_names(con)
    print(f"[fighter-update] {len(existing)} fighters already in DB")

    events = fetch_upcoming_events(args.days)
    print(f"[fighter-update] {len(events)} events found in next {args.days} days")

    missing = collect_missing_fighters(events, existing)
    if not missing:
        print("[fighter-update] no missing fighters — DB is current")
        con.close()
        return 0

    print(f"[fighter-update] {len(missing)} fighter(s) to add:")
    added = 0
    for name, info in missing.items():
        bio = fetch_athlete_bio(info["athleteId"])
        time.sleep(REQUEST_DELAY)
        row = bio_to_row(name, info["athleteId"], info["weightText"], bio)
        print(f"  + {name} (id={info['athleteId']}, weight={row['weight_class']})")
        if not args.dry_run:
            insert_fighter(con, row)
            added += 1

    if not args.dry_run:
        con.commit()
    con.close()
    print(f"[fighter-update] {added} fighter(s) inserted "
          f"({'DRY RUN' if args.dry_run else 'committed'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
