"""Demo data seeding endpoint.

POST /demo/seed — populates a user profile with rich sample data
across all memory stores so users can explore June's features
without building up history organically.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from fastapi import APIRouter
from june_brain.memory import MemoryManager
from pydantic import BaseModel, Field

router = APIRouter(tags=["demo"])


class DemoSeedRequest(BaseModel):
    user_id: str = Field(default="demo", description="User ID to seed data into.")


def _now() -> str:
    return datetime.now().isoformat()


def seed_demo_data(user_id: str) -> dict[str, int]:
    mgr = MemoryManager(user_id)
    counts: dict[str, int] = {}

    seed_goals(mgr, counts)
    seed_calendar(mgr, counts)
    seed_journal(mgr, counts)
    seed_open_loops(mgr, counts)
    seed_body_metrics(mgr, counts)
    seed_facts(mgr, counts)
    seed_entities(mgr, counts)

    return counts


def _inc(counts: dict[str, int], key: str) -> None:
    counts[key] = counts.get(key, 0) + 1


def seed_goals(mgr: MemoryManager, counts: dict[str, int]) -> None:
    goals = [
        ("Run a half marathon", "Fitness", _days(120), "Follow Nike Run Club plan, 4 runs/week", "active"),
        ("Learn Spanish conversationally", "Learning", _days(180), "Duolingo daily + weekly italki tutor", "active"),
        ("Build a side-project portfolio", "Career", _days(90), "Ship MVP of recipe-tracker app", "active"),
        ("Read 24 books this year", "Personal", "", "2 books/month — reading 'Atomic Habits'", "active"),
        ("Meditate daily", "Wellness", "", "10 min every morning using Headspace", "active"),
    ]
    for title, cat, tgt, step, status in goals:
        r = mgr.write({"kind": "goal", "fields": {"title": title, "category": cat, "target_date": tgt, "next_step": step, "status": status}}, source="demo_seed")
        if r.get("written"):
            _inc(counts, "goals")


def seed_calendar(mgr: MemoryManager, counts: dict[str, int]) -> None:
    items = [
        ("Team standup", _days(1), "09:30", "Weekly sync with product team", "planned"),
        ("Gym session", _days(1), "18:00", "Upper body push day", "planned"),
        ("Coffee with mentor", _days(3), "15:00", "Discuss career growth at Blue Bottle", "planned"),
        ("Submit quarterly review", _days(10), "", "Self-assessment and OKR progress due", "planned"),
        ("Flight to Berlin", _days(45), "07:45", "Tech conference — pack light", "planned"),
    ]
    for title, dt, tm, details, status in items:
        r = mgr.write({"kind": "calendar", "fields": {"title": title, "date": dt, "time": tm, "details": details, "status": status}}, source="demo_seed")
        if r.get("written"):
            _inc(counts, "calendar")


def seed_journal(mgr: MemoryManager, counts: dict[str, int]) -> None:
    entries = [
        ("Great run today — 5K in 26:12, a new PB. Felt strong on the hills.", 1),
        ("Finished auth flow with Supabase for the side project. Data model is next.", 2),
        ("Spanish lesson went well. Can order food and talk about hobbies now.", 3),
        ("Meditated 15 minutes this morning on gratitude. Started the day better.", 5),
        ("Read 30 pages of Atomic Habits. Habit stacking is eye-opening.", 7),
    ]
    for text, days_ago in entries:
        day = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")
        r = mgr.write({"kind": "journal", "fields": {"entry": text, "date": day}}, source="demo_seed")
        if r.get("written"):
            _inc(counts, "journal")


def seed_open_loops(mgr: MemoryManager, counts: dict[str, int]) -> None:
    loops = [
        ("Reply about weekend hike", "Check weather and confirm trail with Alex", _days(2), "open"),
        ("Research standing desk converters", "Compare Flexispot vs Jarvis vs Uplift", _days(7), "open"),
        ("Book dentist appointment", "Call clinic to schedule annual checkup", _days(14), "open"),
    ]
    for topic, step, due, status in loops:
        r = mgr.write({"kind": "open_loop", "fields": {"topic": topic, "next_step": step, "due_date": due, "status": status}}, source="demo_seed")
        if r.get("written"):
            _inc(counts, "open_loops")


def seed_body_metrics(mgr: MemoryManager, counts: dict[str, int]) -> None:
    for i in range(5):
        day = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
        r = mgr.write({
            "kind": "body_metric",
            "fields": {
                "date": day,
                "weight_kg": round(78.5 - i * 0.1, 1),
                "sleep_hours": round(7 + (i % 3) * 0.5, 1),
                "energy": 4 if i % 2 == 0 else 3,
                "sleep_quality": 4,
                "steps": 7500 + i * 200,
                "notes": "Feeling good" if i % 2 == 0 else "A bit tired",
            },
        }, source="demo_seed")
        if r.get("written"):
            _inc(counts, "body_metrics")


def seed_facts(mgr: MemoryManager, counts: dict[str, int]) -> None:
    facts = [
        "The demo user prefers dark mode and a clean workspace.",
        "Demo user works as a software engineer focused on full-stack development.",
        "Demo user lives in a city with good running trails nearby.",
        "Demo user is vegetarian and enjoys cooking Asian cuisine.",
        "Demo user is most productive between 8 AM and noon.",
        "Demo user drinks flat white coffee with oat milk.",
        "Demo user listens to lo-fi hip hop while working.",
        "Demo user has a cat named Mochi.",
        "Demo user's favorite book is 'Project Hail Mary' by Andy Weir.",
        "Demo user wants to visit Japan in the next two years.",
    ]
    for text in facts:
        r = mgr.write({"kind": "fact", "fields": {"text": text, "metadata": {"kind": "fact", "source": "demo_seed"}}}, source="demo_seed")
        if r.get("written"):
            _inc(counts, "facts")


def seed_entities(mgr: MemoryManager, counts: dict[str, int]) -> None:
    entities = [
        ("Alex", "person", "Close friend, hiking buddy, works in design"),
        ("Mentor Sarah", "person", "Engineering director, meets monthly for coffee"),
        ("Mochi", "pet", "2-year-old rescue cat, gray tabby"),
        ("Blue Bottle Coffee", "place", "Favorite coffee shop, good for meetings"),
        ("Riverside Trail", "place", "5K running loop along the river"),
        ("Recipe Tracker", "project", "Side project MVP — recipe management app"),
    ]
    for name, kind, desc in entities:
        r = mgr.write({"kind": "entity", "fields": {"label": name, "kind": kind, "props": {"description": desc}}}, source="demo_seed")
        if r.get("written"):
            _inc(counts, "entities")


@router.post("/demo/seed")
def seed_demo(payload: DemoSeedRequest) -> dict:
    """Populate a user profile with rich demo data across all memory stores.

    Creates goals, calendar events, journal entries, open loops, body
    metrics, semantic facts, and entity nodes. After seeding, switch to
    this user in Settings and explore /memory, /skills, and the chat.
    """
    counts = seed_demo_data(payload.user_id)
    return {"user_id": payload.user_id, "seeded": counts}


def _days(n: int) -> str:
    return (datetime.now() + timedelta(days=n)).strftime("%Y-%m-%d")
