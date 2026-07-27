"""June research skill — web search.

Uses Brave Search's free API when ``BRAVE_SEARCH_API_KEY`` is set, otherwise
falls back to DuckDuckGo's HTML endpoint so the skill still works with zero
configuration. Results are summarized to the top ``count`` entries with
title, URL, and snippet.
"""

from __future__ import annotations

import os
import re
import urllib.parse
from html import unescape
from typing import Any

import httpx
from june_brain.guard.ssrf import SsrfBlocked, fetch_guarded
from june_brain.skills.server import MCPStdioServer

server = MCPStdioServer(name="june-research", version="0.1.0")

_BRAVE_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
_DDG_ENDPOINT = "https://duckduckgo.com/html/"


@server.tool(
    name="web_search",
    description=(
        "Search the live web for current information. Use this for weather, "
        "news, prices, or anything the user asks about that would not be in "
        "their personal memory. Returns the top results with title, URL, and "
        "snippet."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query in natural language.",
            },
            "count": {
                "type": "integer",
                "description": "Number of results to return (1-10).",
                "default": 5,
            },
        },
        "required": ["query"],
    },
)
def web_search(query: str, count: int = 5) -> str:
    query = (query or "").strip()
    if not query:
        return "Empty query."
    count = max(1, min(int(count or 5), 10))

    api_key = os.environ.get("BRAVE_SEARCH_API_KEY", "").strip()
    if api_key:
        try:
            return _brave_search(query, count, api_key)
        except Exception as exc:  # noqa: BLE001
            return f"Brave Search failed ({exc}); try DuckDuckGo fallback next time."

    return _ddg_search(query, count)


@server.tool(
    name="fetch_url",
    description=(
        "Fetch a URL and return its cleaned text content. Use when the user "
        "asks about a specific page or when a search result URL looks worth "
        "reading in full."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "Fully qualified http(s) URL to fetch.",
            },
            "max_chars": {
                "type": "integer",
                "description": "Maximum characters of cleaned text to return.",
                "default": 4000,
            },
        },
        "required": ["url"],
    },
)
def fetch_url(url: str, max_chars: int = 4000) -> str:
    url = (url or "").strip()
    if not url.startswith(("http://", "https://")):
        return "URL must start with http:// or https://"
    max_chars = max(500, min(int(max_chars or 4000), 20000))
    # Destination guard (ADR 0021): the content that comes back is framed and
    # scanned, but nothing used to check *where* June was pointed. Redirects are
    # followed one hop at a time so a public URL cannot 302 into localhost.
    try:
        response = fetch_guarded(
            httpx,
            url,
            timeout=15.0,
            headers={"User-Agent": _USER_AGENT},
        )
        response.raise_for_status()
    except SsrfBlocked as exc:
        return f"Refused to fetch {url}: {exc}"
    except Exception as exc:  # noqa: BLE001
        return f"Failed to fetch {url}: {exc}"
    text = _strip_html(response.text)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n[truncated]"
    return f"# {url}\n\n{text}"


# ---------------------------------------------------------------------- helpers


_USER_AGENT = "JuneAI/0.1 (+https://github.com/IrgenSlj/JuneAI)"


def _brave_search(query: str, count: int, api_key: str) -> str:
    response = httpx.get(
        _BRAVE_ENDPOINT,
        params={"q": query, "count": count},
        headers={
            "Accept": "application/json",
            "X-Subscription-Token": api_key,
            "User-Agent": _USER_AGENT,
        },
        timeout=10.0,
    )
    response.raise_for_status()
    payload: dict[str, Any] = response.json()
    results = (payload.get("web") or {}).get("results") or []
    if not results:
        return f"No results for '{query}'."
    lines = [f"Top {min(count, len(results))} results for '{query}' (Brave):"]
    for item in results[:count]:
        title = item.get("title", "").strip()
        snippet = (item.get("description") or "").strip()
        link = item.get("url", "").strip()
        lines.append(f"- {title}\n  {link}\n  {snippet}")
    return "\n".join(lines)


def _ddg_search(query: str, count: int) -> str:
    # Not routed through the SSRF guard on purpose: the destination is a module
    # constant, not user input, so there is nothing for a caller to redirect.
    # Redirects are followed because DuckDuckGo uses them normally.
    try:
        response = httpx.post(
            _DDG_ENDPOINT,
            data={"q": query},
            headers={"User-Agent": _USER_AGENT},
            follow_redirects=True,
            timeout=10.0,
        )
        response.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        return f"DuckDuckGo fallback failed: {exc}"
    items = _parse_ddg_html(response.text, count)
    if not items:
        return f"No results for '{query}'."
    lines = [f"Top {len(items)} results for '{query}' (DuckDuckGo):"]
    for title, link, snippet in items:
        lines.append(f"- {title}\n  {link}\n  {snippet}")
    return "\n".join(lines)


_DDG_RESULT_RE = re.compile(
    r'<a[^>]+class="result__a"[^>]+href="(?P<href>[^"]+)"[^>]*>(?P<title>.*?)</a>'
    r'.*?<a[^>]+class="result__snippet"[^>]*>(?P<snippet>.*?)</a>',
    re.DOTALL,
)


def _parse_ddg_html(html: str, count: int) -> list[tuple[str, str, str]]:
    out: list[tuple[str, str, str]] = []
    for match in _DDG_RESULT_RE.finditer(html):
        href = _unwrap_ddg_redirect(match.group("href"))
        title = _strip_html(match.group("title")).strip()
        snippet = _strip_html(match.group("snippet")).strip()
        if title and href:
            out.append((title, href, snippet))
        if len(out) >= count:
            break
    return out


def _unwrap_ddg_redirect(href: str) -> str:
    # DuckDuckGo wraps results in /l/?uddg=<encoded-url> — decode to the real URL.
    if href.startswith("/l/?") or href.startswith("//duckduckgo.com/l/?"):
        parsed = urllib.parse.urlparse(href)
        qs = urllib.parse.parse_qs(parsed.query)
        if "uddg" in qs:
            return urllib.parse.unquote(qs["uddg"][0])
    if href.startswith("//"):
        return f"https:{href}"
    return href


_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")


def _strip_html(html: str) -> str:
    text = _TAG_RE.sub(" ", html)
    text = unescape(text)
    return _WHITESPACE_RE.sub(" ", text).strip()


def main() -> None:
    server.run()
