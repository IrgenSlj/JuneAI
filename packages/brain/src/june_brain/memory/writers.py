"""Typed write paths, vector paraphrase sync, and delete/update mechanics for MemoryManager (S3 decomposition)."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from datetime import datetime
from hashlib import sha256
from typing import Any

from .paraphrase import (
    _paraphrase_body_metric,
    _paraphrase_calendar,
    _paraphrase_goal,
    _paraphrase_journal,
    _paraphrase_mood,
    _paraphrase_open_loop,
)

logger = logging.getLogger(__name__)


def _vector_fact_id(ref: str) -> str:
    digest = sha256(ref.encode("utf-8")).hexdigest()[:32]
    return f"structured-{digest}"


def write_fact(mgr: Any, fields: dict[str, Any], source: str) -> dict[str, Any]:
    text = str(fields.get("text", "")).strip()
    if not text:
        return {"written": False, "kind": "fact", "ref": None, "stores": []}
    metadata = dict(fields.get("metadata") or {})
    metadata.setdefault("kind", "fact")
    record = mgr.vector.upsert(text=text, source=source, metadata=metadata)
    return {
        "written": True,
        "kind": "fact",
        "ref": f"semantic:{record['fact_id']}",
        "stores": ["vector"],
    }


def write_entity(mgr: Any, fields: dict[str, Any], source: str) -> dict[str, Any]:
    label = str(fields.get("label", "")).strip()
    if not label:
        return {"written": False, "kind": "entity", "ref": None, "stores": []}
    kind = str(fields.get("kind", "entity")).strip() or "entity"
    props = dict(fields.get("props") or {})
    node_id = fields.get("node_id")
    node = mgr.graph.add_node(
        label=label,
        kind=kind,
        props=props,
        **({"node_id": node_id} if node_id else {}),
    )
    return {
        "written": True,
        "kind": "entity",
        "ref": f"node:{node['node_id']}",
        "stores": ["graph"],
        "node_id": node["node_id"],
    }


def write_relation(mgr: Any, fields: dict[str, Any], source: str) -> dict[str, Any]:
    src = str(fields.get("src", "")).strip()
    dst = str(fields.get("dst", "")).strip()
    kind = str(fields.get("kind", "related_to")).strip() or "related_to"
    if not src or not dst:
        return {"written": False, "kind": "relation", "ref": None, "stores": []}
    props = dict(fields.get("props") or {})
    mgr.graph.add_edge(src=src, dst=dst, kind=kind, props=props)
    return {
        "written": True,
        "kind": "relation",
        "ref": f"edge:{src}|{dst}|{kind}",
        "stores": ["graph"],
    }


def write_structured(
    mgr: Any,
    kind: str,
    fields: dict[str, Any],
    source: str,
    *,
    save: Callable[[], dict[str, Any]],
    ref_for: Callable[[dict[str, Any]], str],
    paraphrase: Callable[[dict[str, Any]], str],
) -> dict[str, Any]:
    """Common path: persist the structured row, then paraphrase to vector.

    The paraphrased fact carries ``kind`` and ``ref`` in metadata so
    recall can attribute the hit and the UI can render a back-link.
    Vector upsert failures don't fail the write — the structured row
    is the source of truth; the paraphrase is the recall convenience.
    """
    row = save()
    ref = ref_for(row)
    text = paraphrase(row).strip()
    stores = ["sqlite"]
    if sync_structured_vector(mgr, kind, ref, text, source):
        stores.append("vector")
    return {"written": True, "kind": kind, "ref": ref, "stores": stores}


def sync_structured_vector(
    mgr: Any,
    kind: str,
    ref: str,
    text: str,
    source: str,
) -> bool:
    """Replace the vector paraphrase for one structured memory ref."""
    text = text.strip()
    try:
        mgr.vector.delete_by_ref(ref)
        if not text:
            return False
        mgr.vector.upsert(
            text=text,
            source=source,
            metadata={"kind": kind, "ref": ref},
            fact_id=_vector_fact_id(ref),
        )
        return True
    except Exception:  # noqa: BLE001
        logger.exception("memory.write: vector paraphrase sync failed for %s", ref)
        return False


def delete_structured_vector(mgr: Any, ref: str) -> int:
    try:
        return mgr.vector.delete_by_ref(ref)
    except Exception:  # noqa: BLE001
        logger.exception("memory.forget: vector paraphrase delete failed for %s", ref)
        return 0


def delete_structured_vector_prefix(mgr: Any, ref_prefix: str) -> int:
    try:
        return mgr.vector.delete_by_ref_prefix(ref_prefix)
    except Exception:  # noqa: BLE001
        logger.exception(
            "memory.forget: vector paraphrase prefix delete failed for %s",
            ref_prefix,
        )
        return 0


def write_goal(mgr: Any, fields: dict[str, Any], source: str) -> dict[str, Any]:
    title = str(fields.get("title", "")).strip()
    if not title:
        return {"written": False, "kind": "goal", "ref": None, "stores": []}
    return write_structured(
        mgr,
        "goal",
        fields,
        source,
        save=lambda: mgr.sqlite.save_goal(
            title=title,
            category=str(fields.get("category", "personal")),
            target_date=str(fields.get("target_date", "")),
            next_step=str(fields.get("next_step", "")),
            status=str(fields.get("status", "active")),
        ),
        ref_for=lambda row: f"goal:{row.get('title', '')}",
        paraphrase=lambda row: _paraphrase_goal(row),
    )


def write_open_loop(mgr: Any, fields: dict[str, Any], source: str) -> dict[str, Any]:
    topic = str(fields.get("topic", "")).strip()
    if not topic:
        return {"written": False, "kind": "open_loop", "ref": None, "stores": []}
    return write_structured(
        mgr,
        "open_loop",
        fields,
        source,
        save=lambda: mgr.sqlite.save_open_loop(
            topic=topic,
            next_step=str(fields.get("next_step", "")),
            due_date=str(fields.get("due_date", "")),
            status=str(fields.get("status", "open")),
        ),
        ref_for=lambda row: f"open_loop:{row.get('topic', '')}",
        paraphrase=lambda row: _paraphrase_open_loop(row),
    )


def write_calendar(mgr: Any, fields: dict[str, Any], source: str) -> dict[str, Any]:
    title = str(fields.get("title", "")).strip()
    date = str(fields.get("date", "")).strip()
    if not title:
        return {"written": False, "kind": "calendar", "ref": None, "stores": []}
    return write_structured(
        mgr,
        "calendar",
        fields,
        source,
        save=lambda: mgr.sqlite.save_calendar_item(
            title=title,
            date=date,
            time=str(fields.get("time", "")),
            details=str(fields.get("details", "")),
            status=str(fields.get("status", "planned")),
            source=str(fields.get("source", "conversation")),
        ),
        ref_for=lambda row: f"calendar:{row.get('title', '')}|{row.get('date', '')}|{row.get('time', '')}",
        paraphrase=lambda row: _paraphrase_calendar(row),
    )


def write_journal(mgr: Any, fields: dict[str, Any], source: str) -> dict[str, Any]:
    entry = str(fields.get("entry", "")).strip()
    if not entry:
        return {"written": False, "kind": "journal", "ref": None, "stores": []}
    # save_journal returns {entry, timestamp} but not the auto-id; fetch
    # the most recent to get it for the ref.
    mgr.sqlite.save_journal(entry)
    latest = mgr.sqlite.get_journal(limit=1)
    if not latest:
        return {"written": False, "kind": "journal", "ref": None, "stores": ["sqlite"]}
    row = latest[0]
    ref = f"journal:{row.get('id', '')}"
    text = _paraphrase_journal(row)
    stores = ["sqlite"]
    if sync_structured_vector(mgr, "journal", ref, text, source):
        stores.append("vector")
    return {"written": True, "kind": "journal", "ref": ref, "stores": stores}


def write_body_metric(mgr: Any, fields: dict[str, Any], source: str) -> dict[str, Any]:
    return write_structured(
        mgr,
        "body_metric",
        fields,
        source,
        save=lambda: mgr.sqlite.log_body_metrics(
            weight_kg=float(fields.get("weight_kg") or 0),
            sleep_hours=float(fields.get("sleep_hours") or 0),
            sleep_quality=int(fields.get("sleep_quality") or 0),
            energy=int(fields.get("energy") or 0),
            stress=int(fields.get("stress") or 0),
            soreness=int(fields.get("soreness") or 0),
            resting_hr=int(fields.get("resting_hr") or 0),
            steps=int(fields.get("steps") or 0),
            notes=str(fields.get("notes", "")),
            date=str(fields.get("date", "")),
        ),
        ref_for=lambda row: f"body_metric:{row.get('date', '')}",
        paraphrase=lambda row: _paraphrase_body_metric(row),
    )


def write_mood(mgr: Any, fields: dict[str, Any], source: str) -> dict[str, Any]:
    mood = str(fields.get("mood", "")).strip()
    if not mood:
        return {"written": False, "kind": "mood", "ref": None, "stores": []}
    note = str(fields.get("note", ""))
    # Mood rows are append-only; the timestamp returned by log_mood is
    # the stable identifier for any future ref-based lookup.
    row = mgr.sqlite.log_mood(mood, note)
    ref = f"mood:{row.get('timestamp', '')}"
    text = _paraphrase_mood(row)
    stores = ["sqlite"]
    if sync_structured_vector(mgr, "mood", ref, text, source):
        stores.append("vector")
    return {"written": True, "kind": "mood", "ref": ref, "stores": stores}


def list_forgotten_all(mgr: Any, limit: int) -> list[dict[str, Any]]:
    """Merge per-store trash bins into one ref-keyed list, most recently forgotten first."""
    entries: list[dict[str, Any]] = []
    for f in mgr.vector.list_forgotten(limit=limit):
        entries.append({
            "ref": f"semantic:{f['fact_id']}", "kind": "fact", "text": f["text"],
            "created_at": f.get("created_at", ""), "forgotten_at": f.get("forgotten_at", ""),
        })
    for n in mgr.graph.list_forgotten_nodes(limit=limit):
        entries.append({
            "ref": f"node:{n['node_id']}", "kind": n.get("kind") or "entity",
            "text": n["label"], "created_at": n.get("updated_at", ""),
            "forgotten_at": n.get("forgotten_at", ""),
        })
    for s in mgr.sqlite.list_forgotten_structured(limit=limit):
        entries.append({
            "ref": s["ref"], "kind": s["kind"], "text": s["summary"],
            "created_at": "", "forgotten_at": s["forgotten_at"],
        })
    entries.sort(key=lambda e: e["forgotten_at"], reverse=True)
    return entries[:limit]


def _archive_structured(mgr: Any, ref: str, kind: str) -> bool:
    """Snapshot a structured row into forgotten_structured before deletion.

    Returns True if a row was found and archived, False if not found.
    The archive INSERT and the live-row delete are separate commits (the
    existing delete helpers commit internally). Archive-then-delete keeps
    things correct: a crash between archive and delete leaves a spurious
    trash entry the user can purge, never a lost row.
    """
    conn = mgr.sqlite._conn
    user_id = mgr.sqlite.user_id
    now = datetime.now().isoformat()

    row: dict | None = None
    fields: dict
    summary: str

    if kind == "goal":
        title = ref.removeprefix("goal:")
        r = conn.execute(
            "SELECT title,category,target_date,next_step,status FROM goals "
            "WHERE user_id=? AND lower(title)=lower(?)",
            (user_id, title),
        ).fetchone()
        if r is None:
            return False
        row = dict(r)
        fields = {k: row[k] for k in ("title", "category", "target_date", "next_step", "status")}
        summary = row["title"]

    elif kind == "open_loop":
        topic = ref.removeprefix("open_loop:")
        r = conn.execute(
            "SELECT topic,next_step,due_date,status FROM open_loops "
            "WHERE user_id=? AND lower(topic)=lower(?)",
            (user_id, topic),
        ).fetchone()
        if r is None:
            return False
        row = dict(r)
        fields = {k: row[k] for k in ("topic", "next_step", "due_date", "status")}
        summary = row["topic"]

    elif kind == "calendar":
        body = ref.removeprefix("calendar:")
        parts = body.split("|", 2)
        title = parts[0]
        date = parts[1] if len(parts) > 1 else ""
        time = parts[2] if len(parts) > 2 else ""
        query = (
            "SELECT title,date,time,details,status,source FROM calendar_items "
            "WHERE user_id=? AND lower(title)=lower(?)"
        )
        params: list = [user_id, title]
        if date:
            query += " AND lower(date)=lower(?)"
            params.append(date)
        if time:
            query += " AND lower(time)=lower(?)"
            params.append(time)
        r = conn.execute(query, params).fetchone()
        if r is None:
            return False
        row = dict(r)
        # row["source"] is the calendar-event origin (e.g. "conversation");
        # write_calendar reads it via fields.get("source", "conversation").
        fields = {k: row[k] for k in ("title", "date", "time", "details", "status", "source")}
        summary = row["title"]

    elif kind == "journal":
        entry_id_str = ref.removeprefix("journal:")
        try:
            entry_id = int(entry_id_str)
        except ValueError:
            return False
        r = conn.execute(
            "SELECT id,entry FROM journal WHERE user_id=? AND id=?",
            (user_id, entry_id),
        ).fetchone()
        if r is None:
            return False
        row = dict(r)
        fields = {"entry": row["entry"]}
        summary = row["entry"][:80]

    elif kind == "body_metric":
        date = ref.removeprefix("body_metric:")
        r = conn.execute(
            "SELECT date,weight_kg,sleep_hours,sleep_quality,energy,stress,"
            "soreness,resting_hr,steps,notes FROM body_metrics "
            "WHERE user_id=? AND date=?",
            (user_id, date),
        ).fetchone()
        if r is None:
            return False
        row = dict(r)
        fields = {k: row[k] for k in (
            "date", "weight_kg", "sleep_hours", "sleep_quality", "energy",
            "stress", "soreness", "resting_hr", "steps", "notes",
        )}
        summary = f"Body check on {row['date']}"

    else:
        return False

    with conn:
        conn.execute(
            """INSERT INTO forgotten_structured
                 (user_id, ref, kind, summary, fields, source, forgotten_at)
               VALUES (?,?,?,?,?,?,?)
               ON CONFLICT(user_id, ref) DO UPDATE SET
                 kind=excluded.kind, summary=excluded.summary,
                 fields=excluded.fields, source=excluded.source,
                 forgotten_at=excluded.forgotten_at""",
            (user_id, ref, kind, summary, json.dumps(fields), "manual", now),
        )
    return True


def restore_structured(mgr: Any, ref: str) -> dict | None:
    """Restore a structured row from the forgotten_structured trash.

    Replays the original write via mgr.write so the live row AND its
    vector paraphrase are both recreated through the same tested code path.
    Deletes the trash entry after a successful write.
    Returns the write-result dict, or None if the ref is not in the trash.
    """
    conn = mgr.sqlite._conn
    user_id = mgr.sqlite.user_id
    r = conn.execute(
        "SELECT ref,kind,fields,source FROM forgotten_structured "
        "WHERE user_id=? AND ref=?",
        (user_id, ref),
    ).fetchone()
    if r is None:
        return None
    kind = r["kind"]
    fields = json.loads(r["fields"])
    source = r["source"]
    result = mgr.write({"kind": kind, "fields": fields}, source)
    if not result.get("written"):
        return None
    conn.execute(
        "DELETE FROM forgotten_structured WHERE user_id=? AND ref=?",
        (user_id, ref),
    )
    conn.commit()
    return result


def forget(mgr: Any, ref: str) -> bool:
    """Remove a fact by its ``ref`` (as returned from recall hits).

    Refs are prefixed with the source so we know which store to hit:
      - "semantic:<fact_id>" → vector + shadow table
      - "node:<node_id>"     → graph node (and all its edges)
      - "edge:<src>|<dst>|<kind>" → single graph edge
      - "goal:<title>"       → SQLite goals row
      - "open_loop:<topic>"  → SQLite open_loops row
      - "calendar:<title>" or "calendar:<title>|<date>|<time>"
                             → SQLite calendar_items row
      - "journal:<id>"       → SQLite journal row
      - "body_metric:<date>" → SQLite body_metrics row
      - anything else falls through to the vector store by raw id
        (so the memory-browser UI can pass a fact_id directly).
    """
    ref = ref.strip()
    if not ref:
        return False
    if ref.startswith("semantic:"):
        fact_id = ref.removeprefix("semantic:")
        # Reversible: archive to the trash, not a hard delete (vision / CLAUDE.md).
        return mgr.vector.forget(fact_id)
    if ref.startswith("node:"):
        node_id = ref.removeprefix("node:")
        # Reversible: archive the entity to the trash, not a hard delete.
        return mgr.graph.forget_node(node_id)
    if ref.startswith("edge:"):
        body = ref.removeprefix("edge:")
        parts = body.split("|", 2)
        if len(parts) == 3:
            mgr.graph.remove_edge(parts[0], parts[1], parts[2])
            return True
        return False
    if ref.startswith("goal:"):
        title = ref.removeprefix("goal:")
        _archive_structured(mgr, ref, "goal")
        removed_sqlite = mgr.sqlite.delete_goal(title)
        removed_vector = delete_structured_vector(mgr, ref)
        return removed_sqlite or removed_vector > 0
    if ref.startswith("open_loop:"):
        topic = ref.removeprefix("open_loop:")
        _archive_structured(mgr, ref, "open_loop")
        removed_sqlite = mgr.sqlite.delete_open_loop(topic)
        removed_vector = delete_structured_vector(mgr, ref)
        return removed_sqlite or removed_vector > 0
    if ref.startswith("calendar:"):
        body = ref.removeprefix("calendar:")
        parts = body.split("|", 2)
        title = parts[0]
        date = parts[1] if len(parts) > 1 else ""
        time = parts[2] if len(parts) > 2 else ""
        _archive_structured(mgr, ref, "calendar")
        removed_sqlite = mgr.sqlite.delete_calendar_item(title, date, time)
        if len(parts) > 1:
            removed_vector = delete_structured_vector(mgr, ref)
        else:
            removed_vector = delete_structured_vector(mgr, ref)
            removed_vector += delete_structured_vector_prefix(mgr, f"{ref}|")
        return removed_sqlite or removed_vector > 0
    if ref.startswith("journal:"):
        entry_id = ref.removeprefix("journal:")
        try:
            entry_id_int = int(entry_id)
        except ValueError:
            return False
        _archive_structured(mgr, ref, "journal")
        removed_sqlite = mgr.sqlite.delete_journal_entry(entry_id_int)
        removed_vector = delete_structured_vector(mgr, ref)
        return removed_sqlite or removed_vector > 0
    if ref.startswith("body_metric:"):
        date = ref.removeprefix("body_metric:")
        _archive_structured(mgr, ref, "body_metric")
        removed_sqlite = mgr.sqlite.delete_body_metric(date)
        removed_vector = delete_structured_vector(mgr, ref)
        return removed_sqlite or removed_vector > 0
    # Fall-through: treat as a vector fact_id. Reversible, like semantic: refs.
    return mgr.vector.forget(ref)


def update(
    mgr: Any,
    ref: str,
    fields: dict[str, str],
    source: str = "manual",
) -> dict | None:
    """Patch a structured-row memory by ``ref``.

    Returns the updated row dict (with the *new* fields), or ``None`` if
    the row was not found. Only the SQLite ref kinds are supported here:
    semantic facts and graph nodes have their own edit paths (re-upsert
    and add_node respectively).
    """
    ref = ref.strip()
    if not ref:
        return None
    if ref.startswith("goal:"):
        old_title = ref.removeprefix("goal:")
        row = mgr.sqlite.update_goal(old_title, **fields)
        if row is not None:
            new_ref = f"goal:{row.get('title', '')}"
            if new_ref != ref:
                delete_structured_vector(mgr, ref)
            sync_structured_vector(
                mgr,
                "goal",
                new_ref,
                _paraphrase_goal(row),
                source,
            )
        return row
    if ref.startswith("open_loop:"):
        old_topic = ref.removeprefix("open_loop:")
        row = mgr.sqlite.update_open_loop(old_topic, **fields)
        if row is not None:
            new_ref = f"open_loop:{row.get('topic', '')}"
            if new_ref != ref:
                delete_structured_vector(mgr, ref)
            sync_structured_vector(
                mgr,
                "open_loop",
                new_ref,
                _paraphrase_open_loop(row),
                source,
            )
        return row
    if ref.startswith("calendar:"):
        body = ref.removeprefix("calendar:")
        parts = body.split("|", 2)
        old_title = parts[0]
        old_date = parts[1] if len(parts) > 1 else ""
        old_time = parts[2] if len(parts) > 2 else ""
        row = mgr.sqlite.update_calendar_item(old_title, old_date, old_time, **fields)
        if row is not None:
            new_ref = (
                f"calendar:{row.get('title', '')}|"
                f"{row.get('date', '')}|{row.get('time', '')}"
            )
            if new_ref != ref:
                delete_structured_vector(mgr, ref)
            sync_structured_vector(
                mgr,
                "calendar",
                new_ref,
                _paraphrase_calendar(row),
                source,
            )
        return row
    return None


WRITE_HANDLERS: dict[str, Callable[..., dict[str, Any]]] = {
    "fact": write_fact,
    "entity": write_entity,
    "relation": write_relation,
    "goal": write_goal,
    "open_loop": write_open_loop,
    "calendar": write_calendar,
    "journal": write_journal,
    "body_metric": write_body_metric,
    "mood": write_mood,
}
