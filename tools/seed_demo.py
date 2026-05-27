"""Seed a demo user profile with rich sample data across all memory stores.

Usage:
    python tools/seed_demo.py [--user USER] [--api URL]

Defaults:
    --user  demo
    --api   http://localhost:8000

Or use the API endpoint:
    curl -X POST http://localhost:8000/demo/seed -H "Content-Type: application/json" -d '{"user_id": "demo"}'
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from urllib.request import Request, urlopen

API_URL = "http://localhost:8000"
DEMO_USER = "demo"

SEED_DATA = {
    "goals": [
        {
            "title": "Run a half marathon",
            "category": "Fitness",
            "target_date": (datetime.now() + timedelta(days=120)).strftime("%Y-%m-%d"),
            "next_step": "Follow Nike Run Club half-marathon plan, 4 runs/week",
            "status": "active",
        },
        {
            "title": "Learn Spanish conversationally",
            "category": "Learning",
            "target_date": (datetime.now() + timedelta(days=180)).strftime("%Y-%m-%d"),
            "next_step": "Duolingo daily + weekly italki tutor session",
            "status": "active",
        },
        {
            "title": "Build a side-project portfolio",
            "category": "Career",
            "target_date": (datetime.now() + timedelta(days=90)).strftime("%Y-%m-%d"),
            "next_step": "Ship MVP of recipe-tracker app by end of month",
            "status": "active",
        },
        {
            "title": "Read 24 books this year",
            "category": "Personal",
            "target_date": "",
            "next_step": "2 books/month — currently reading 'Atomic Habits'",
            "status": "active",
        },
        {
            "title": "Meditate daily",
            "category": "Wellness",
            "target_date": "",
            "next_step": "10 min every morning using Headspace",
            "status": "active",
        },
    ],
    "open_loops": [
        {
            "topic": "Reply about weekend hike",
            "next_step": "Check weather forecast and confirm trail with Alex",
            "due_date": (datetime.now() + timedelta(days=2)).strftime("%Y-%m-%d"),
            "status": "open",
        },
        {
            "topic": "Research standing desk converters",
            "next_step": "Compare brands: Flexispot vs Jarvis vs Uplift",
            "due_date": (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d"),
            "status": "open",
        },
        {
            "topic": "Book dentist appointment",
            "next_step": "Call clinic to schedule annual checkup",
            "due_date": (datetime.now() + timedelta(days=14)).strftime("%Y-%m-%d"),
            "status": "open",
        },
    ],
    "calendar": [
        {
            "title": "Team standup",
            "date": (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d"),
            "time": "09:30",
            "details": "Weekly sync with product team",
            "status": "planned",
        },
        {
            "title": "Gym session",
            "date": (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d"),
            "time": "18:00",
            "details": "Upper body push day",
            "status": "planned",
        },
        {
            "title": "Coffee with mentor",
            "date": (datetime.now() + timedelta(days=3)).strftime("%Y-%m-%d"),
            "time": "15:00",
            "details": "Discuss career growth and side projects at Blue Bottle",
            "status": "planned",
        },
        {
            "title": "Submit quarterly review",
            "date": (datetime.now() + timedelta(days=10)).strftime("%Y-%m-%d"),
            "time": "",
            "details": "Self-assessment and OKR progress due",
            "status": "planned",
        },
        {
            "title": "Flight to Berlin",
            "date": (datetime.now() + timedelta(days=45)).strftime("%Y-%m-%d"),
            "time": "07:45",
            "details": "Tech conference — pack light",
            "status": "planned",
        },
    ],
    "journal": [
        {
            "entry": "Great run today — 5K in 26:12, which is a PB. Felt strong on the hills. Going to try for 10K next weekend.",
            "days_ago": 1,
        },
        {
            "entry": "Finished the first chapter of my side project. Got the auth flow working with Supabase. Need to think about the data model for recipes.",
            "days_ago": 2,
        },
        {
            "entry": "Spanish lesson went well. Can now order food and talk about hobbies. Still mixing up 'por' and 'para'.",
            "days_ago": 3,
        },
        {
            "entry": "Meditated for 15 minutes this morning — focused on gratitude. Started the day in a much better headspace.",
            "days_ago": 5,
        },
        {
            "entry": "Read 30 pages of 'Atomic Habits' today. The idea of habit stacking is eye-opening. Going to try: after my morning coffee, I'll meditate for 5 minutes.",
            "days_ago": 7,
        },
    ],
    "body_metrics": [
        {"date": (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d"), "weight_kg": round(78.5 - i * 0.1, 1), "sleep_hours": round(7 + (i % 3) * 0.5, 1), "water_glasses": 6 + (i % 4), "energy": "good" if i % 2 == 0 else "fair"}
        for i in range(7)
    ],
    "semantic_facts": [
        "The demo user prefers dark mode and a clean workspace.",
        "Demo user works in tech as a software engineer focused on full-stack development.",
        "Demo user lives in a city with good running trails nearby.",
        "Demo user is vegetarian and enjoys cooking Asian cuisine.",
        "Demo user is most productive between 8 AM and noon.",
        "Demo user drinks flat white coffee with oat milk.",
        "Demo user listens to lo-fi hip hop while working.",
        "Demo user has a cat named Mochi.",
        "Demo user's favorite book is 'Project Hail Mary' by Andy Weir.",
        "Demo user wants to visit Japan in the next two years.",
    ],
    "entities": [
        {"name": "Alex", "kind": "person", "desc": "Close friend, hiking buddy, works in design"},
        {"name": "Mentor Sarah", "kind": "person", "desc": "Engineering director at former company, meets monthly for coffee"},
        {"name": "Mochi", "kind": "pet", "desc": "2-year-old rescue cat, gray tabby, loves laser pointers"},
        {"name": "Blue Bottle Coffee", "kind": "place", "desc": "Favorite coffee shop, good for meetings"},
        {"name": "Riverside Trail", "kind": "place", "desc": "5K running loop along the river, flat terrain"},
        {"name": "Side Project Inc.", "kind": "project", "desc": "Recipe tracker app — MVP in progress"},
    ],
}


def seed_via_api(api_url: str, user_id: str) -> dict:
    """Send seed data through June's API if available."""
    base = api_url.rstrip("/")

    results = {}

    # Goals
    for g in SEED_DATA["goals"]:
        req = Request(
            f"{base}/memory/{user_id}/fact",
            method="POST",
            data=json.dumps({"kind": "goal", "fields": g}).encode(),
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        try:
            with urlopen(req, timeout=10) as resp:
                results.setdefault("goals", []).append(json.loads(resp.read()))
        except Exception as e:
            results.setdefault("errors", []).append(f"goal '{g['title']}': {e}")

    # Open loops
    for o in SEED_DATA["open_loops"]:
        req = Request(
            f"{base}/memory/{user_id}/fact",
            method="POST",
            data=json.dumps({"kind": "open_loop", "fields": o}).encode(),
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        try:
            with urlopen(req, timeout=10) as resp:
                results.setdefault("open_loops", []).append(json.loads(resp.read()))
        except Exception as e:
            results.setdefault("errors", []).append(f"open_loop '{o['topic']}': {e}")

    # Calendar
    for c in SEED_DATA["calendar"]:
        req = Request(
            f"{base}/memory/{user_id}/fact",
            method="POST",
            data=json.dumps({"kind": "calendar", "fields": c}).encode(),
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        try:
            with urlopen(req, timeout=10) as resp:
                results.setdefault("calendar", []).append(json.loads(resp.read()))
        except Exception as e:
            results.setdefault("errors", []).append(f"calendar '{c['title']}': {e}")

    # Journal
    for j in SEED_DATA["journal"]:
        day = (datetime.now() - timedelta(days=j["days_ago"])).strftime("%Y-%m-%d")
        req = Request(
            f"{base}/memory/{user_id}/fact",
            method="POST",
            data=json.dumps({"kind": "journal", "fields": {"entry": j["entry"], "date": day}}).encode(),
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        try:
            with urlopen(req, timeout=10) as resp:
                results.setdefault("journal", []).append(json.loads(resp.read()))
        except Exception as e:
            results.setdefault("errors", []).append(f"journal: {e}")

    # Body metrics
    for b in SEED_DATA["body_metrics"]:
        req = Request(
            f"{base}/memory/{user_id}/fact",
            method="POST",
            data=json.dumps({"kind": "body_metric", "fields": b}).encode(),
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        try:
            with urlopen(req, timeout=10) as resp:
                results.setdefault("body_metrics", []).append(json.loads(resp.read()))
        except Exception as e:
            results.setdefault("errors", []).append(f"body_metric: {e}")

    # Semantic facts (vector store)
    for text in SEED_DATA["semantic_facts"]:
        req = Request(
            f"{base}/memory/{user_id}/fact",
            method="POST",
            data=json.dumps({"kind": "fact", "fields": {"text": text, "metadata": {"kind": "fact", "source": "demo_seed"}}}).encode(),
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        try:
            with urlopen(req, timeout=10) as resp:
                results.setdefault("facts", []).append(json.loads(resp.read()))
        except Exception as e:
            results.setdefault("errors", []).append(f"fact: {e}")

    # Entities
    for e in SEED_DATA["entities"]:
        req = Request(
            f"{base}/memory/{user_id}/fact",
            method="POST",
            data=json.dumps({"kind": "entity", "fields": {"label": e["name"], "kind": e["kind"], "props": {"description": e["desc"]}}}).encode(),
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        try:
            with urlopen(req, timeout=10) as resp:
                results.setdefault("entities", []).append(json.loads(resp.read()))
        except Exception as e:
            results.setdefault("errors", []).append(f"entity '{e['name']}': {e}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed demo data into a June user profile")
    parser.add_argument("--user", default=DEMO_USER, help="User ID to seed")
    parser.add_argument("--api", default=API_URL, help="June API base URL")
    args = parser.parse_args()

    print(f"Seeding demo data for user '{args.user}' via {args.api} ...")
    results = seed_via_api(args.api, args.user)
    total = sum(len(v) for k, v in results.items() if k != "errors")
    errors = results.get("errors", [])
    print(f"Done — {total} items written")
    if errors:
        print(f"Warnings ({len(errors)}):")
        for e in errors:
            print(f"  {e}")
    print(f"Switch to user '{args.user}' in Settings to explore.")


if __name__ == "__main__":
    main()
