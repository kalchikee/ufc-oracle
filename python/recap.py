#!/usr/bin/env python3
"""UFC recap — grade past predictions against ESPN MMA scoreboard results.

UFC Oracle's recap path previously updated predictions.correct in the SQLite
DB, but the DB lives in an actions/artifact and the recap can fail silently
(historically because UFCStats was JS-challenged; now fixed via ESPN
fallback but the chain is fragile). This script mirrors the WNBA/EPL
pattern: walk every predictions/<date>.json, look up the event on ESPN's
MMA scoreboard for that date + ±2-day window, match each pick to its
actual winner by fighter-name pair, and append to data/grading_history.json.

The morning Discord embed should read grading_history.json instead of the
DB so the season-accuracy field actually populates.

    {
      "graded": [{
        "date": "2026-05-16",
        "eventName": "UFC Fight Night: Allen vs. Costa",
        "gameId": "ufc-2026-05-16-arnold-allen-melquizael-costa",
        "fighterA": "Arnold Allen", "fighterB": "Melquizael Costa",
        "pickedFighter": "Melquizael Costa",
        "modelProb": 0.6009,
        "actualWinner": "Arnold Allen",
        "correct": false,
        "isMainEvent": true,
        "confidenceTier": "strong",
        "gradedAt": "..."
      }]
    }

Idempotent. Re-running on a date with already-graded picks is a no-op.

Usage:
    python python/recap.py                   # grade everything ungraded
    python python/recap.py --date 2026-05-16 # grade just one date
"""
from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).parent.parent
PREDICTIONS_DIR = ROOT / "predictions"
HISTORY_FILE = ROOT / "data" / "grading_history.json"
ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/mma/ufc/scoreboard"

LOOKBACK_DAYS = 2  # how far before/after prediction date to search ESPN


def normalize_name(s: str) -> str:
    """Lower-case + accent-strip so 'Édgar Cháirez' matches 'Edgar Chairez'."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.lower().strip()


def load_history() -> dict:
    if not HISTORY_FILE.exists():
        return {"graded": []}
    try:
        return json.loads(HISTORY_FILE.read_text())
    except Exception as e:
        backup = HISTORY_FILE.with_suffix(
            f".corrupt-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}.json"
        )
        try:
            HISTORY_FILE.rename(backup)
            print(f"[recap] CORRUPT history file — preserved as {backup}: {e}")
        except Exception:
            print(f"[recap] CORRUPT history file — starting fresh: {e}")
        return {"graded": []}


def save_history(history: dict) -> None:
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = HISTORY_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(history, indent=2))
    tmp.replace(HISTORY_FILE)


def fetch_espn_events(iso_date: str) -> list[dict]:
    """Return all UFC events listed by ESPN for the given date."""
    yyyymmdd = iso_date.replace("-", "")
    try:
        r = requests.get(f"{ESPN_BASE}?dates={yyyymmdd}", timeout=10,
                         headers={"User-Agent": "UFC-Oracle-Recap/4.1"})
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"[recap] ESPN fetch failed for {iso_date}: {e}")
        return []
    return data.get("events", []) or []


def find_event_for_date(iso_date: str) -> dict | None:
    """Look up the ESPN event for a prediction date.

    UFC cards are usually Saturday but workflows can predict ±2 days off
    (timezone roll-over on the night-of, plus midweek Fight Nights).
    Search the prediction date and a small window around it; pick the
    event closest to the prediction date."""
    base = datetime.strptime(iso_date, "%Y-%m-%d")
    candidates = []
    for offset in range(-LOOKBACK_DAYS, LOOKBACK_DAYS + 1):
        d = (base + timedelta(days=offset)).strftime("%Y-%m-%d")
        for ev in fetch_espn_events(d):
            ev_date = ev.get("date", "")[:10]
            candidates.append((abs(offset), ev_date, ev))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])  # closest to prediction date first
    return candidates[0][2]


def grade_date(iso_date: str, history: dict) -> int:
    pred_file = PREDICTIONS_DIR / f"{iso_date}.json"
    if not pred_file.exists():
        return 0
    try:
        preds = json.loads(pred_file.read_text())
    except Exception as e:
        print(f"[recap] could not parse {pred_file}: {e}")
        return 0

    picks = preds.get("picks", []) or []
    if not picks:
        return 0

    already = {g["gameId"] for g in history["graded"] if g.get("date") == iso_date}
    event = find_event_for_date(iso_date)
    if not event:
        print(f"[recap] {iso_date}: no ESPN event found in window")
        return 0

    # Build a lookup keyed by (normalized name pair) -> winner name.
    # We try both orderings since pickedSide and ESPN's home/away differ.
    bouts: dict[frozenset, dict] = {}
    for c in event.get("competitions", []) or []:
        if not c.get("status", {}).get("type", {}).get("completed"):
            continue
        competitors = c.get("competitors", []) or []
        if len(competitors) < 2:
            continue
        a = competitors[0].get("athlete", {})
        b = competitors[1].get("athlete", {})
        a_name = a.get("displayName") or ""
        b_name = b.get("displayName") or ""
        if not a_name or not b_name:
            continue
        winner_obj = next(
            (x for x in competitors if x.get("winner")), None,
        )
        winner_name = (winner_obj or {}).get("athlete", {}).get("displayName")
        if not winner_name:
            continue
        key = frozenset({normalize_name(a_name), normalize_name(b_name)})
        bouts[key] = {
            "fighterA": a_name,
            "fighterB": b_name,
            "winner": winner_name,
            "winnerNorm": normalize_name(winner_name),
        }

    if not bouts:
        print(f"[recap] {iso_date}: event found but no completed bouts")
        return 0

    newly = 0
    for pick in picks:
        gid = pick.get("gameId")
        if gid in already:
            continue
        # Predictions JSON has top-level home/away for the bout; pickedTeam
        # is the predicted winner.
        ha = pick.get("home") or ""
        aw = pick.get("away") or ""
        picked = pick.get("pickedTeam") or ""
        key = frozenset({normalize_name(ha), normalize_name(aw)})
        match = bouts.get(key)
        if not match:
            print(f"[recap] {iso_date} {ha} vs {aw}: not found in ESPN bouts")
            continue
        correct = normalize_name(picked) == match["winnerNorm"]
        extra = pick.get("extra") or {}
        history["graded"].append({
            "date": iso_date,
            "eventName": extra.get("eventName") or event.get("name"),
            "gameId": gid,
            "fighterA": ha,
            "fighterB": aw,
            "pickedFighter": picked,
            "modelProb": pick.get("modelProb"),
            "actualWinner": match["winner"],
            "correct": correct,
            "isMainEvent": bool(extra.get("isMainEvent")),
            "confidenceTier": pick.get("confidenceTier"),
            "gradedAt": datetime.now(timezone.utc).isoformat(),
        })
        newly += 1
    return newly


def grade_all_ungraded(history: dict) -> int:
    if not PREDICTIONS_DIR.exists():
        return 0
    total = 0
    today_iso = datetime.now().strftime("%Y-%m-%d")
    for f in sorted(PREDICTIONS_DIR.glob("*.json")):
        iso = f.stem
        if iso >= today_iso:
            continue
        total += grade_date(iso, history)
    return total


# Confidence buckets shown in the Discord embed so the user can see how
# the model's declared probability tracks reality at each tier. Each
# bucket is half-open [lo, hi). For UFC the recap writes pick.modelProb
# directly into grading_history.json — that value is ALREADY the
# picked-side probability (the prob the model gave to the predicted
# winner), so the lowest possible value is ~0.5 and we do NOT take
# max(p, 1-p).
CONFIDENCE_BUCKETS = [
    (0.50, 0.60, "50-60%"),
    (0.60, 0.70, "60-70%"),
    (0.70, 0.80, "70-80%"),
    (0.80, 0.90, "80-90%"),
    (0.90, 1.01, "90%+"),
]


def compute_confidence_buckets(history: dict, year: int | None = None) -> list[dict]:
    """For each confidence bucket return {label, total, correct, accuracy}.

    Only buckets with at least one graded pick are returned, so the
    embed doesn't carry empty rows when the season is young."""
    graded = history.get("graded", [])
    if year is not None:
        graded = [g for g in graded if g.get("date", "").startswith(str(year))]
    out = []
    for lo, hi, label in CONFIDENCE_BUCKETS:
        rows = [g for g in graded
                if g.get("modelProb") is not None
                and lo <= float(g["modelProb"]) < hi]
        if not rows:
            continue
        correct = sum(1 for r in rows if r.get("correct"))
        out.append({
            "label":    label,
            "total":    len(rows),
            "correct":  correct,
            "accuracy": correct / len(rows),
        })
    return out


def compute_season_stats(history: dict, year: int | None = None) -> dict:
    """Year-to-date if year is given, else lifetime. Matches the
    AccuracyStats shape that getYTDAccuracy returns in TypeScript."""
    graded = history.get("graded", [])
    if year is not None:
        graded = [g for g in graded if g.get("date", "").startswith(str(year))]
    total = len(graded)
    correct = sum(1 for g in graded if g.get("correct"))
    hc = [g for g in graded if g.get("confidenceTier") in ("high_conviction", "lock")]
    hc_correct = sum(1 for g in hc if g.get("correct"))
    main = [g for g in graded if g.get("isMainEvent")]
    main_correct = sum(1 for g in main if g.get("correct"))
    return {
        "total": total,
        "correct": correct,
        "accuracy": (correct / total) if total > 0 else 0.0,
        "hcTotal": len(hc),
        "hcCorrect": hc_correct,
        "hcAccuracy": (hc_correct / len(hc)) if hc else 0.0,
        "mainEventTotal": len(main),
        "mainEventCorrect": main_correct,
        "mainEventAccuracy": (main_correct / len(main)) if main else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", help="YYYY-MM-DD; grade only this date")
    args = parser.parse_args()

    history = load_history()
    if args.date:
        iso = args.date if "-" in args.date else (
            f"{args.date[:4]}-{args.date[4:6]}-{args.date[6:8]}")
        newly = grade_date(iso, history)
    else:
        newly = grade_all_ungraded(history)

    if newly > 0:
        save_history(history)

    year = datetime.now().year
    stats = compute_season_stats(history, year)
    print(f"[recap] newly graded: {newly} picks")
    print(f"[recap] {year} season: {stats['correct']}/{stats['total']} correct "
          f"({stats['accuracy']*100:.1f}%)")
    print(f"[recap] {year} HC: {stats['hcCorrect']}/{stats['hcTotal']} "
          f"({stats['hcAccuracy']*100:.1f}%)")
    print(f"[recap] {year} main events: {stats['mainEventCorrect']}"
          f"/{stats['mainEventTotal']} ({stats['mainEventAccuracy']*100:.1f}%)")

    buckets = compute_confidence_buckets(history, year)
    if buckets:
        print(f"[recap] {year} calibration by confidence:")
        for b in buckets:
            print(f"  {b['label']}: {b['correct']}/{b['total']} "
                  f"({b['accuracy']*100:.1f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
