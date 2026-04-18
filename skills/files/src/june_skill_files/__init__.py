"""June files skill — local PDFs and webpage extraction.

``read_pdf`` uses pypdf to extract text from a file on disk. ``read_webpage``
uses trafilatura to pull the readable body of an article or page, stripping
navigation, comments, and ads.

Paths are resolved relative to the user's HOME for safety — callers cannot
escape to system files. A whitelisted set of absolute roots can be added
later via env if needed.
"""

from __future__ import annotations

from pathlib import Path

import httpx

from june_brain.skills.server import MCPStdioServer

server = MCPStdioServer(name="june-files", version="0.1.0")


@server.tool(
    name="read_pdf",
    description=(
        "Read a PDF file from disk and return its extracted text. Path can be "
        "absolute or relative to the user's home directory."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "File path to a .pdf document.",
            },
            "max_chars": {
                "type": "integer",
                "description": "Maximum characters of text to return.",
                "default": 8000,
            },
        },
        "required": ["path"],
    },
)
def read_pdf(path: str, max_chars: int = 8000) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        return "pypdf is not installed — reinstall the files skill."

    resolved = _resolve_user_path(path)
    if resolved is None:
        return f"Refusing to read {path!r}: not inside the user's home directory."
    if not resolved.exists():
        return f"File not found: {resolved}"
    if resolved.suffix.lower() != ".pdf":
        return f"Not a PDF: {resolved.name}"

    max_chars = max(500, min(int(max_chars or 8000), 40000))
    try:
        reader = PdfReader(str(resolved))
        chunks: list[str] = []
        total = 0
        for index, page in enumerate(reader.pages, start=1):
            try:
                text = (page.extract_text() or "").strip()
            except Exception as exc:  # noqa: BLE001
                text = f"[page {index} extraction failed: {exc}]"
            if not text:
                continue
            chunks.append(f"## Page {index}\n{text}")
            total += len(text)
            if total >= max_chars:
                chunks.append("\n[truncated]")
                break
        if not chunks:
            return f"{resolved.name}: PDF contained no extractable text."
        return f"# {resolved.name}\n\n" + "\n\n".join(chunks)
    except Exception as exc:  # noqa: BLE001
        return f"Failed to read {resolved}: {exc}"


@server.tool(
    name="read_webpage",
    description=(
        "Fetch a URL and return its readable content (article body, stripped "
        "of nav/comments). Use when the user wants the contents of a page "
        "rather than a search summary."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "Fully qualified http(s) URL.",
            },
            "max_chars": {
                "type": "integer",
                "description": "Maximum characters of cleaned text to return.",
                "default": 6000,
            },
        },
        "required": ["url"],
    },
)
def read_webpage(url: str, max_chars: int = 6000) -> str:
    url = (url or "").strip()
    if not url.startswith(("http://", "https://")):
        return "URL must start with http:// or https://"
    max_chars = max(500, min(int(max_chars or 6000), 30000))

    try:
        response = httpx.get(
            url,
            follow_redirects=True,
            timeout=15.0,
            headers={"User-Agent": "JuneAI/0.1 (+files-skill)"},
        )
        response.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        return f"Failed to fetch {url}: {exc}"

    try:
        import trafilatura
    except ImportError:
        return "trafilatura is not installed — reinstall the files skill."

    extracted = trafilatura.extract(
        response.text,
        include_comments=False,
        include_tables=True,
        favor_precision=True,
    )
    if not extracted:
        return f"Could not extract readable content from {url}."
    if len(extracted) > max_chars:
        extracted = extracted[:max_chars] + "\n\n[truncated]"
    return f"# {url}\n\n{extracted}"


def _resolve_user_path(path: str) -> Path | None:
    """Resolve a user-supplied path relative to ``$HOME`` and confirm containment."""
    home = Path.home().resolve()
    raw = Path(path).expanduser()
    target = raw if raw.is_absolute() else home / raw
    try:
        resolved = target.resolve()
    except (OSError, RuntimeError):
        return None
    try:
        resolved.relative_to(home)
    except ValueError:
        return None
    return resolved


def main() -> None:
    server.run()
