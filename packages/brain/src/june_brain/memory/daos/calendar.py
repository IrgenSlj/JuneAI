"""Calendar DAO — calendar_items table."""

from datetime import date, datetime, timezone

_now = lambda: datetime.now(timezone.utc).isoformat()
_today = lambda: date.today()


def _parse_date(value: str) -> date | None:
    try:
        return date.fromisoformat(value.strip())
    except (TypeError, ValueError, AttributeError):
        return None


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS calendar_items (
    user_id     TEXT NOT NULL,
    title       TEXT NOT NULL,
    date        TEXT NOT NULL DEFAULT '',
    time        TEXT NOT NULL DEFAULT '',
    details     TEXT NOT NULL DEFAULT '',
    status      TEXT NOT NULL DEFAULT 'planned',
    source      TEXT NOT NULL DEFAULT 'conversation',
    updated_at  TEXT NOT NULL,
    PRIMARY KEY (user_id, title, date, time)
);
"""


class CalendarDAO:
    def __init__(self, conn, user_id: str):
        self._conn = conn
        self.user_id = user_id

    def save_calendar_item(
        self,
        title: str,
        date: str,
        time: str = "",
        details: str = "",
        status: str = "planned",
        source: str = "conversation",
    ) -> dict:
        item = {
            "title": title.strip(),
            "date": date.strip(),
            "time": time.strip(),
            "details": details.strip(),
            "status": status.strip() or "planned",
            "source": source.strip() or "conversation",
            "updated_at": _now(),
        }
        self._conn.execute(
            """INSERT INTO calendar_items (user_id,title,date,time,details,status,source,updated_at)
               VALUES (?,?,?,?,?,?,?,?)
               ON CONFLICT(user_id,title,date,time) DO UPDATE SET
                 details=excluded.details, status=excluded.status,
                 source=excluded.source, updated_at=excluded.updated_at""",
            (self.user_id, item["title"], item["date"], item["time"],
             item["details"], item["status"], item["source"], item["updated_at"]),
        )
        self._conn.commit()
        return item

    def update_calendar_item_status(
        self,
        title: str,
        status: str,
        date: str = "",
        time: str = "",
    ) -> dict | None:
        now = _now()
        query = "UPDATE calendar_items SET status=?, updated_at=? WHERE user_id=? AND lower(title)=lower(?)"
        params: list = [status.strip() or "planned", now, self.user_id, title.strip()]
        if date.strip():
            query += " AND lower(date)=lower(?)"
            params.append(date.strip())
        if time.strip():
            query += " AND lower(time)=lower(?)"
            params.append(time.strip())
        self._conn.execute(query, params)
        self._conn.commit()
        row = self._conn.execute(
            "SELECT title,date,time,details,status,source,updated_at FROM calendar_items "
            "WHERE user_id=? AND lower(title)=lower(?)",
            (self.user_id, title.strip()),
        ).fetchone()
        return dict(row) if row else None

    def update_calendar_item(
        self,
        old_title: str,
        old_date: str = "",
        old_time: str = "",
        **fields: str,
    ) -> dict | None:
        query = (
            "SELECT title,date,time,details,status,source,updated_at "
            "FROM calendar_items WHERE user_id=? AND lower(title)=lower(?)"
        )
        params: list = [self.user_id, old_title.strip()]
        if old_date.strip():
            query += " AND lower(date)=lower(?)"
            params.append(old_date.strip())
        if old_time.strip():
            query += " AND lower(time)=lower(?)"
            params.append(old_time.strip())
        rows = self._conn.execute(query, params).fetchall()
        if not rows:
            return None
        existing = dict(rows[0])
        merged = {**existing, **{k: v for k, v in fields.items() if v is not None}}
        new_title = (merged.get("title") or "").strip() or existing["title"]
        new_date = (merged.get("date") or "").strip()
        new_time = (merged.get("time") or "").strip()
        pk_changed = (
            new_title.lower() != existing["title"].lower()
            or new_date.lower() != existing["date"].lower()
            or new_time.lower() != existing["time"].lower()
        )
        if pk_changed:
            self.delete_calendar_item(existing["title"], existing["date"], existing["time"])
        return self.save_calendar_item(
            title=new_title,
            date=new_date,
            time=new_time,
            details=merged.get("details", ""),
            status=merged.get("status", "planned"),
            source=merged.get("source", "conversation"),
        )

    def delete_calendar_item(self, title: str, date: str = "", time: str = "") -> bool:
        query = "DELETE FROM calendar_items WHERE user_id=? AND lower(title)=lower(?)"
        params: list = [self.user_id, title.strip()]
        if date.strip():
            query += " AND lower(date)=lower(?)"
            params.append(date.strip())
        if time.strip():
            query += " AND lower(time)=lower(?)"
            params.append(time.strip())
        cur = self._conn.execute(query, params)
        self._conn.commit()
        return cur.rowcount > 0

    def get_calendar_items(self, status: str = "", limit: int = 20) -> list:
        if status.strip():
            rows = self._conn.execute(
                "SELECT title,date,time,details,status,source,updated_at FROM calendar_items "
                "WHERE user_id=? AND lower(status)=lower(?)",
                (self.user_id, status.strip()),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT title,date,time,details,status,source,updated_at FROM calendar_items "
                "WHERE user_id=?",
                (self.user_id,),
            ).fetchall()
        items = [dict(r) for r in rows]
        today = _today()
        dated, undated = [], []
        for item in items:
            parsed = _parse_date(item.get("date", ""))
            if parsed is None:
                undated.append(item)
            else:
                dated.append((parsed, item))
        dated.sort(key=lambda e: (abs((e[0] - today).days), e[0], e[1].get("time", ""), e[1].get("title", "")))
        undated.sort(key=lambda i: (i.get("title", ""), i.get("updated_at", "")))
        return ([i for _, i in dated] + undated)[:limit]
