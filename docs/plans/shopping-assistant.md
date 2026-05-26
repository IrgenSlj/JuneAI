# Shopping Assistant Plan

> **Status:** Backlog feature plan. Shopping should not ship as a standalone
> vertical before the v0.1.1 capture, intent, approval, event-ledger, and Daily
> Home foundation is in place. When revived, shopping inputs enter through the
> same quick-capture pipeline as every other domain.

## Overview

A personal shopping assistant that tracks products the user wants, remembers their preferences and budget, monitors prices, and provides proactive deal alerts.

## Architecture

Shopping is implemented as:
1. **New domain tables** in the SQLite memory store (`products`, `purchase_history`, `price_alerts`)
2. **New agent tools** for capturing products, checking prices, setting alerts
3. **Scheduled price checks** via the Scheduler Service (Phase 1 Component 1)
4. **Integration with shopping APIs** via MCP skills (future: Amazon, BestBuy, etc.)

## Domain Schema

```sql
CREATE TABLE products (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    name TEXT NOT NULL,
    category TEXT NOT NULL DEFAULT 'general',
    preferred_price REAL,
    preferred_store TEXT DEFAULT '',
    notes TEXT DEFAULT '',
    url TEXT DEFAULT '',
    date_added TEXT NOT NULL,
    active INTEGER DEFAULT 1
);

CREATE TABLE purchase_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    product_id INTEGER NOT NULL REFERENCES products(id),
    price REAL,
    store TEXT DEFAULT '',
    date TEXT NOT NULL,
    notes TEXT DEFAULT ''
);

CREATE TABLE price_alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    product_id INTEGER NOT NULL REFERENCES products(id),
    target_price REAL NOT NULL,
    active INTEGER DEFAULT 1,
    created_at TEXT NOT NULL
);
```

## Agent Tools

### Capture & Tracking
- `track_product(name, category, preferred_price, store, notes, url)` — add a product to the watchlist
- `log_purchase(product_name, price, store, notes)` — record a purchase
- `list_products(category, active_only)` — view tracked products
- `update_product(product_id, **fields)` — update product details
- `remove_product(product_id)` — remove from tracking

### Price Alerts
- `set_price_alert(product_name, target_price)` — get notified when price drops below threshold
- `list_price_alerts()` — view active alerts
- `remove_price_alert(alert_id)` — remove an alert

### Proactive
- `get_shopping_summary()` — overview of tracked products, pending purchases, active alerts
- `check_price_drops()` — (scheduled) check if any watched products changed price (requires external API)

## Memory DAO

New file: `packages/brain/src/june_brain/memory/dao_shopping.py`

```python
class ShoppingDAO:
    def __init__(self, user_id: str, conn: sqlite3.Connection):
        self.user_id = user_id
        self.conn = conn
    
    def add_product(self, name, category, preferred_price, store, notes, url) -> dict
    def get_products(self, category=None, active_only=True) -> list[dict]
    def update_product(self, product_id, **fields) -> dict | None
    def remove_product(self, product_id) -> bool
    
    def log_purchase(self, product_id, price, store, notes) -> dict
    def get_purchase_history(self, product_id=None, limit=50) -> list[dict]
    
    def set_price_alert(self, product_id, target_price) -> dict
    def get_price_alerts(self, active_only=True) -> list[dict]
    def remove_price_alert(self, alert_id) -> bool
```

## Implementation Order

### Session S1 — Schema + DAO
- [ ] Add `products`, `purchase_history`, `price_alerts` tables to schema DDL
- [ ] Create alembic migration for new tables
- [ ] Implement `ShoppingDAO` class
- [ ] Wire into `Memory` facade (or `MemoryManager` write path)

### Session S2 — Agent Tools
- [ ] Implement all shopping agent tools in `tools.py`
- [ ] Add to `JUNE_TOOLS` list
- [ ] Test each tool with mock data

### Session S3 — UI
- [ ] Shopping page in SvelteKit (`/shopping` or section in `/memory`)
- [ ] Product list with search/filter
- [ ] Add product form
- [ ] Price alert management UI
- [ ] Purchase history view

### Session S4 — Proactive Features
- [ ] Scheduled price check (skeleton — checks nothing real without API)
- [ ] Notification when price alert triggers
- [ ] Weekly shopping summary in daily briefing
- [ ] "You haven't bought X in a while — still interested?" nudge
