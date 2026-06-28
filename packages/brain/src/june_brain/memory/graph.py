"""Knowledge graph for entity relationships.

People, places, projects, and recurring concepts are nodes. Relationships
between them are typed edges. Storage is SQLite tables (``graph_nodes``
and ``graph_edges``) in the same ``june.db`` so the user's brain stays
portable as a single file.

The graph is what lets June say "you mentioned Ana three times this
week" or "you tend to reach out to Marco when you're stressed." It is
the relational layer that the structured store (schemas) and semantic
store (embeddings) cannot cover on their own.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any

from .sqlite import _get_connection, db_path


def _slug(value: str) -> str:
    """Turn free text into a stable identifier."""
    cleaned = re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")
    return cleaned or "entity"


def _now() -> str:
    return datetime.now().isoformat()


def _db_path() -> str:
    return db_path()


class KnowledgeGraph:
    """Typed entity graph stored in SQLite, scoped per user."""

    def __init__(self, user_id: str) -> None:
        self.user_id = user_id

    @property
    def _conn(self):
        return _get_connection(_db_path())

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def add_node(
        self,
        label: str,
        kind: str = "entity",
        props: dict[str, Any] | None = None,
        node_id: str | None = None,
    ) -> dict[str, Any]:
        """Create or update a node. Uses kind:slug(label) as the default id."""
        label = label.strip()
        if not label:
            raise ValueError("node label cannot be empty")
        kind = kind.strip() or "entity"
        node_id = node_id or f"{kind}:{_slug(label)}"
        props_json = json.dumps(props or {})
        now = _now()
        self._conn.execute(
            """INSERT INTO graph_nodes (user_id, node_id, kind, label, props, updated_at)
               VALUES (?,?,?,?,?,?)
               ON CONFLICT(user_id, node_id) DO UPDATE SET
                 kind=excluded.kind, label=excluded.label,
                 props=excluded.props, updated_at=excluded.updated_at""",
            (self.user_id, node_id, kind, label, props_json, now),
        )
        self._conn.commit()
        return {
            "node_id": node_id,
            "kind": kind,
            "label": label,
            "props": props or {},
            "updated_at": now,
        }

    def add_edge(
        self,
        src: str,
        dst: str,
        kind: str = "related_to",
        props: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create or update a directed edge from ``src`` to ``dst``."""
        src = src.strip()
        dst = dst.strip()
        if not src or not dst:
            raise ValueError("edge endpoints cannot be empty")
        kind = kind.strip() or "related_to"
        props_json = json.dumps(props or {})
        now = _now()
        self._conn.execute(
            """INSERT INTO graph_edges (user_id, src, dst, kind, props, updated_at)
               VALUES (?,?,?,?,?,?)
               ON CONFLICT(user_id, src, dst, kind) DO UPDATE SET
                 props=excluded.props, updated_at=excluded.updated_at""",
            (self.user_id, src, dst, kind, props_json, now),
        )
        self._conn.commit()
        return {
            "src": src,
            "dst": dst,
            "kind": kind,
            "props": props or {},
            "updated_at": now,
        }

    def remove_node(self, node_id: str) -> None:
        """Delete a node and all edges touching it."""
        self._conn.execute(
            "DELETE FROM graph_edges WHERE user_id=? AND (src=? OR dst=?)",
            (self.user_id, node_id, node_id),
        )
        self._conn.execute(
            "DELETE FROM graph_nodes WHERE user_id=? AND node_id=?",
            (self.user_id, node_id),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Reversible forget — archive an entity before deleting it, mirroring the
    # vector store's trash so forgetting stays conservative and reversible.
    # Edges touching the node are not restored (they may point at other
    # already-gone nodes); the entity itself returns with its original id.
    # ------------------------------------------------------------------

    def forget_node(self, node_id: str) -> bool:
        """Archive a node to the trash, then remove it (and its edges)."""
        node = self.get_node(node_id)
        if node is None:
            return False
        self._conn.execute(
            """INSERT INTO forgotten_nodes
                 (user_id, node_id, kind, label, props, updated_at, forgotten_at)
               VALUES (?,?,?,?,?,?,?)
               ON CONFLICT(user_id, node_id) DO UPDATE SET
                 kind=excluded.kind, label=excluded.label, props=excluded.props,
                 updated_at=excluded.updated_at, forgotten_at=excluded.forgotten_at""",
            (
                self.user_id,
                node["node_id"],
                node["kind"],
                node["label"],
                json.dumps(node["props"]),
                node["updated_at"],
                _now(),
            ),
        )
        self._conn.commit()
        self.remove_node(node_id)
        return True

    def list_forgotten_nodes(self, limit: int = 50) -> list[dict[str, Any]]:
        """List trashed nodes, most recently forgotten first."""
        rows = self._conn.execute(
            "SELECT node_id, kind, label, props, updated_at, forgotten_at "
            "FROM forgotten_nodes WHERE user_id=? ORDER BY forgotten_at DESC LIMIT ?",
            (self.user_id, limit),
        ).fetchall()
        return [
            {
                "node_id": r["node_id"],
                "kind": r["kind"],
                "label": r["label"],
                "props": _loads(r["props"]),
                "updated_at": r["updated_at"],
                "forgotten_at": r["forgotten_at"],
            }
            for r in rows
        ]

    def purge_forgotten_nodes(self) -> int:
        """Permanently empty the entity trash. Returns the number of rows removed."""
        cur = self._conn.execute(
            "DELETE FROM forgotten_nodes WHERE user_id=?", (self.user_id,)
        )
        self._conn.commit()
        return int(cur.rowcount or 0)

    def restore_node(self, node_id: str) -> dict[str, Any] | None:
        """Bring a trashed node back to the live graph with its original id.

        Atomic: the node re-insert and the trash-row delete happen in one
        transaction, so a crash can never leave the entity in both stores. The
        original ``updated_at`` is preserved so a restored entity does not jump
        to the top of recency-ordered views.
        """
        conn = self._conn
        row = conn.execute(
            "SELECT node_id, kind, label, props, updated_at FROM forgotten_nodes "
            "WHERE user_id=? AND node_id=?",
            (self.user_id, node_id),
        ).fetchone()
        if not row:
            return None
        with conn:  # single transaction: commit both writes or neither
            conn.execute(
                """INSERT INTO graph_nodes (user_id, node_id, kind, label, props, updated_at)
                   VALUES (?,?,?,?,?,?)
                   ON CONFLICT(user_id, node_id) DO UPDATE SET
                     kind=excluded.kind, label=excluded.label,
                     props=excluded.props, updated_at=excluded.updated_at""",
                (
                    self.user_id,
                    row["node_id"],
                    row["kind"],
                    row["label"],
                    row["props"],
                    row["updated_at"],
                ),
            )
            conn.execute(
                "DELETE FROM forgotten_nodes WHERE user_id=? AND node_id=?",
                (self.user_id, node_id),
            )
        return {
            "node_id": row["node_id"],
            "kind": row["kind"],
            "label": row["label"],
            "props": _loads(row["props"]),
            "updated_at": row["updated_at"],
        }

    def remove_edge(self, src: str, dst: str, kind: str = "") -> None:
        """Delete an edge. If ``kind`` is empty, remove all edges between the two nodes."""
        if kind.strip():
            self._conn.execute(
                "DELETE FROM graph_edges WHERE user_id=? AND src=? AND dst=? AND kind=?",
                (self.user_id, src, dst, kind.strip()),
            )
        else:
            self._conn.execute(
                "DELETE FROM graph_edges WHERE user_id=? AND src=? AND dst=?",
                (self.user_id, src, dst),
            )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def get_node(self, node_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT node_id, kind, label, props, updated_at FROM graph_nodes "
            "WHERE user_id=? AND node_id=?",
            (self.user_id, node_id),
        ).fetchone()
        return self._row_to_node(row) if row else None

    def find_nodes(
        self,
        kind: str | None = None,
        query: str = "",
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Return nodes, optionally filtered by kind and a substring match on label."""
        sql = "SELECT node_id, kind, label, props, updated_at FROM graph_nodes WHERE user_id=?"
        params: list[Any] = [self.user_id]
        if kind:
            sql += " AND kind=?"
            params.append(kind.strip())
        if query.strip():
            sql += " AND lower(label) LIKE ?"
            params.append(f"%{query.strip().lower()}%")
        sql += " ORDER BY updated_at DESC LIMIT ?"
        params.append(limit)
        rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_node(r) for r in rows]

    def neighbors(
        self,
        node_id: str,
        kind: str | None = None,
        direction: str = "both",
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Return neighbors of ``node_id`` along with the edge that connects them."""
        rows: list[Any] = []
        if direction in ("out", "both"):
            sql = (
                "SELECT e.src, e.dst, e.kind AS edge_kind, e.props AS edge_props, "
                "       n.node_id, n.kind AS node_kind, n.label, n.props AS node_props, n.updated_at "
                "FROM graph_edges e JOIN graph_nodes n "
                "  ON n.user_id=e.user_id AND n.node_id=e.dst "
                "WHERE e.user_id=? AND e.src=?"
            )
            params: list[Any] = [self.user_id, node_id]
            if kind:
                sql += " AND e.kind=?"
                params.append(kind.strip())
            sql += " ORDER BY e.updated_at DESC LIMIT ?"
            params.append(limit)
            rows.extend(self._conn.execute(sql, params).fetchall())
        if direction in ("in", "both"):
            sql = (
                "SELECT e.src, e.dst, e.kind AS edge_kind, e.props AS edge_props, "
                "       n.node_id, n.kind AS node_kind, n.label, n.props AS node_props, n.updated_at "
                "FROM graph_edges e JOIN graph_nodes n "
                "  ON n.user_id=e.user_id AND n.node_id=e.src "
                "WHERE e.user_id=? AND e.dst=?"
            )
            params = [self.user_id, node_id]
            if kind:
                sql += " AND e.kind=?"
                params.append(kind.strip())
            sql += " ORDER BY e.updated_at DESC LIMIT ?"
            params.append(limit)
            rows.extend(self._conn.execute(sql, params).fetchall())

        seen: set[str] = set()
        results = []
        for r in rows:
            key = f"{r['src']}|{r['dst']}|{r['edge_kind']}"
            if key in seen:
                continue
            seen.add(key)
            results.append(
                {
                    "edge": {
                        "src": r["src"],
                        "dst": r["dst"],
                        "kind": r["edge_kind"],
                        "props": _loads(r["edge_props"]),
                    },
                    "node": {
                        "node_id": r["node_id"],
                        "kind": r["node_kind"],
                        "label": r["label"],
                        "props": _loads(r["node_props"]),
                        "updated_at": r["updated_at"],
                    },
                }
            )
        return results[:limit]

    def mentions_near(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """Find nodes whose label appears in ``query`` (case-insensitive word match).

        Used during recall to surface entities the user is likely referring
        to — cheap, string-based entry point into the graph.
        """
        query_text = query.strip().lower()
        if not query_text:
            return []
        nodes = self.find_nodes(limit=200)
        matches = []
        for node in nodes:
            label = node["label"].lower()
            if not label:
                continue
            if re.search(rf"\b{re.escape(label)}\b", query_text):
                matches.append(node)
        matches.sort(key=lambda n: n.get("updated_at", ""), reverse=True)
        return matches[:limit]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_node(row: Any) -> dict[str, Any]:
        return {
            "node_id": row["node_id"],
            "kind": row["kind"],
            "label": row["label"],
            "props": _loads(row["props"]),
            "updated_at": row["updated_at"],
        }


def _loads(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    try:
        result = json.loads(value)
        return result if isinstance(result, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}
