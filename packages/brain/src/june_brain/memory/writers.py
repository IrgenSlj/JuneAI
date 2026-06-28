"""Typed write paths, vector paraphrase sync, and delete/update mechanics for MemoryManager (S3 decomposition)."""

from __future__ import annotations

import logging
from collections.abc import Callable
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
        if not mgr.graph.get_node(node_id):
            return False
        mgr.graph.remove_node(node_id)
        return True
    if ref.startswith("edge:"):
        body = ref.removeprefix("edge:")
        parts = body.split("|", 2)
        if len(parts) == 3:
            mgr.graph.remove_edge(parts[0], parts[1], parts[2])
            return True
        return False
    if ref.startswith("goal:"):
        title = ref.removeprefix("goal:")
        removed_sqlite = mgr.sqlite.delete_goal(title)
        removed_vector = delete_structured_vector(mgr, ref)
        return removed_sqlite or removed_vector > 0
    if ref.startswith("open_loop:"):
        topic = ref.removeprefix("open_loop:")
        removed_sqlite = mgr.sqlite.delete_open_loop(topic)
        removed_vector = delete_structured_vector(mgr, ref)
        return removed_sqlite or removed_vector > 0
    if ref.startswith("calendar:"):
        body = ref.removeprefix("calendar:")
        parts = body.split("|", 2)
        title = parts[0]
        date = parts[1] if len(parts) > 1 else ""
        time = parts[2] if len(parts) > 2 else ""
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
            removed_sqlite = mgr.sqlite.delete_journal_entry(int(entry_id))
        except ValueError:
            return False
        removed_vector = delete_structured_vector(mgr, ref)
        return removed_sqlite or removed_vector > 0
    if ref.startswith("body_metric:"):
        date = ref.removeprefix("body_metric:")
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
