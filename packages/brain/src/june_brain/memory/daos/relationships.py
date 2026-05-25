"""Relationships DAO — relationships table."""

from datetime import datetime, timezone

_now = lambda: datetime.now(timezone.utc).isoformat()

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS relationships (
    user_id         TEXT NOT NULL,
    person          TEXT NOT NULL,
    relationship    TEXT NOT NULL DEFAULT '',
    summary         TEXT NOT NULL DEFAULT '',
    user_needs      TEXT NOT NULL DEFAULT '',
    cautions        TEXT NOT NULL DEFAULT '',
    updated_at      TEXT NOT NULL,
    PRIMARY KEY (user_id, person)
);
"""


class RelationshipDAO:
    def __init__(self, conn, user_id: str):
        self._conn = conn
        self.user_id = user_id

    def save_relationship_profile(
        self,
        person: str,
        relationship: str,
        summary: str,
        user_needs: str = "",
        cautions: str = "",
    ) -> dict:
        item = {
            "person": person.strip(),
            "relationship": relationship.strip(),
            "summary": summary.strip(),
            "user_needs": user_needs.strip(),
            "cautions": cautions.strip(),
            "updated_at": _now(),
        }
        self._conn.execute(
            """INSERT INTO relationships (user_id,person,relationship,summary,user_needs,cautions,updated_at)
               VALUES (?,?,?,?,?,?,?)
               ON CONFLICT(user_id,person) DO UPDATE SET
                 relationship=excluded.relationship, summary=excluded.summary,
                 user_needs=excluded.user_needs, cautions=excluded.cautions,
                 updated_at=excluded.updated_at""",
            (self.user_id, item["person"], item["relationship"], item["summary"],
             item["user_needs"], item["cautions"], item["updated_at"]),
        )
        self._conn.commit()
        return item

    def get_relationship_profiles(self, person: str = "") -> list:
        if person.strip():
            rows = self._conn.execute(
                "SELECT person,relationship,summary,user_needs,cautions,updated_at "
                "FROM relationships WHERE user_id=? AND lower(person)=lower(?)",
                (self.user_id, person.strip()),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT person,relationship,summary,user_needs,cautions,updated_at "
                "FROM relationships WHERE user_id=?",
                (self.user_id,),
            ).fetchall()
        return [dict(r) for r in rows]
