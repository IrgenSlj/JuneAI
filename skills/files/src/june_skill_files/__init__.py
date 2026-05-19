"""June files skill — sandboxed filesystem access plus webpage extraction.

The skill exposes five tools:

- ``list_directory`` — list the contents of a folder under the user's HOME.
- ``read_file`` — read a UTF-8 text file (any extension) under the user's HOME.
- ``search_files`` — find files by filename glob or by content match.
- ``read_pdf`` — extract text from a PDF on disk.
- ``read_webpage`` — fetch and clean the readable body of a URL.

Every path-taking tool resolves the input relative to ``$HOME`` and refuses
anything that resolves outside it (including symlink traversal). This is the
brain's first sandboxing primitive; OS-level Tauri permission grants come in
a later slice but the policy here matches what the brain will request.
"""

from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import Any

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


_LIST_LIMIT = 200
_SEARCH_LIMIT = 100
_CONTENT_MAX_CHARS = 200_000
_SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build", ".next"}
_TEXT_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".rst",
    ".py", ".js", ".ts", ".tsx", ".jsx", ".svelte", ".vue",
    ".html", ".htm", ".css", ".scss", ".sass",
    ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf",
    ".csv", ".tsv",
    ".sh", ".bash", ".zsh", ".fish",
    ".sql", ".graphql", ".proto",
    ".c", ".h", ".cpp", ".hpp", ".cs", ".go", ".rs", ".java", ".kt", ".swift", ".rb", ".php",
    ".log", ".env",
}


@server.tool(
    name="list_directory",
    description=(
        "List the contents of a folder under the user's HOME directory. Returns "
        "file and subdirectory names with sizes. Pass 'show_hidden=true' to "
        "include dotfiles."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Directory path. Absolute or relative to home; '~' or empty means home.",
            },
            "show_hidden": {
                "type": "boolean",
                "description": "Include dotfiles and dot-directories.",
                "default": False,
            },
        },
        "required": ["path"],
    },
)
def list_directory(path: str, show_hidden: bool = False) -> str:
    resolved = _resolve_user_path(path or ".")
    if resolved is None:
        return f"Refusing to list {path!r}: not inside the user's home directory."
    if not resolved.exists():
        return f"Directory not found: {resolved}"
    if not resolved.is_dir():
        return f"Not a directory: {resolved}"

    home = Path.home().resolve()
    rel_display = _display_rel(resolved, home)

    entries: list[tuple[str, str, int]] = []  # (kind, name, size)
    try:
        children = sorted(resolved.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    except PermissionError:
        return f"Permission denied: {rel_display}"

    for child in children:
        name = child.name
        if not show_hidden and name.startswith("."):
            continue
        try:
            if child.is_dir():
                entries.append(("dir", name, -1))
            else:
                entries.append(("file", name, child.stat().st_size))
        except (OSError, PermissionError):
            entries.append(("?", name, -1))
        if len(entries) >= _LIST_LIMIT:
            entries.append(("…", f"(truncated at {_LIST_LIMIT})", -1))
            break

    if not entries:
        return f"# {rel_display}\n\n(empty)"

    lines = [f"# {rel_display}", ""]
    for kind, name, size in entries:
        if kind == "dir":
            lines.append(f"  [dir]  {name}/")
        elif kind == "file":
            lines.append(f"  [file] {name}  ({_format_size(size)})")
        else:
            lines.append(f"  [{kind}]    {name}")
    return "\n".join(lines)


@server.tool(
    name="read_file",
    description=(
        "Read a UTF-8 text file from disk under the user's HOME directory. For "
        "PDFs, use read_pdf instead. Returns up to max_chars characters."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "File path. Absolute or relative to home.",
            },
            "max_chars": {
                "type": "integer",
                "description": "Maximum characters to return.",
                "default": 16000,
            },
        },
        "required": ["path"],
    },
)
def read_file(path: str, max_chars: int = 16000) -> str:
    resolved = _resolve_user_path(path)
    if resolved is None:
        return f"Refusing to read {path!r}: not inside the user's home directory."
    if not resolved.exists():
        return f"File not found: {resolved}"
    if not resolved.is_file():
        return f"Not a file: {resolved}"

    if resolved.suffix.lower() == ".pdf":
        return f"{resolved.name} is a PDF — use read_pdf instead."

    max_chars = max(500, min(int(max_chars or 16000), _CONTENT_MAX_CHARS))
    try:
        try:
            text = resolved.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return f"{resolved.name} is not valid UTF-8 text. Use read_pdf for PDFs."
    except (OSError, PermissionError) as exc:
        return f"Failed to read {resolved}: {exc}"

    home = Path.home().resolve()
    rel = _display_rel(resolved, home)
    truncated = ""
    if len(text) > max_chars:
        text = text[:max_chars]
        truncated = f"\n\n[truncated at {max_chars} chars]"
    return f"# {rel}\n\n{text}{truncated}"


@server.tool(
    name="search_files",
    description=(
        "Search for files by filename glob and/or by content text. Both modes "
        "scan recursively under the given root (limited to the user's HOME). "
        "Returns matching paths and short context snippets for content matches."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "root": {
                "type": "string",
                "description": "Root folder to search under. Defaults to the user's HOME.",
                "default": "~",
            },
            "name_pattern": {
                "type": "string",
                "description": "Filename glob, e.g. '*.pdf' or 'invoice-*.txt'.",
            },
            "content_query": {
                "type": "string",
                "description": "Substring to find inside text files (case-insensitive).",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum matches to return.",
                "default": 30,
            },
        },
    },
)
def search_files(
    root: str = "~",
    name_pattern: str = "",
    content_query: str = "",
    max_results: int = 30,
) -> str:
    name_pattern = (name_pattern or "").strip()
    content_query = (content_query or "").strip()
    if not name_pattern and not content_query:
        return "Provide name_pattern, content_query, or both."

    base = _resolve_user_path(root or "~")
    if base is None:
        return f"Refusing to search under {root!r}: not inside the user's home directory."
    if not base.exists() or not base.is_dir():
        return f"Search root is not a directory: {base}"

    home = Path.home().resolve()
    max_results = max(1, min(int(max_results or 30), _SEARCH_LIMIT))
    needle = content_query.lower() if content_query else ""

    matches: list[str] = []
    files_scanned = 0

    for path in _walk(base):
        if len(matches) >= max_results:
            break
        if path.name.startswith("."):
            continue
        name_ok = (not name_pattern) or fnmatch.fnmatch(path.name, name_pattern)
        if not name_ok:
            continue

        if not content_query:
            matches.append(f"  {_display_rel(path, home)}")
            continue

        files_scanned += 1
        if path.suffix.lower() not in _TEXT_EXTENSIONS:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except (OSError, PermissionError):
            continue
        lower = text.lower()
        idx = lower.find(needle)
        if idx == -1:
            continue
        snippet = _excerpt(text, idx, len(content_query))
        matches.append(f"  {_display_rel(path, home)}\n    {snippet}")

    if not matches:
        return (
            f"No matches under {_display_rel(base, home)} "
            f"(filename={name_pattern!r}, content={content_query!r})."
        )
    body = "\n".join(matches)
    return f"# Search results under {_display_rel(base, home)}\n\n{body}"


def _walk(root: Path) -> Any:
    """Iterate files under root, skipping noisy dirs and bailing on permission errors."""
    stack: list[Path] = [root]
    while stack:
        current = stack.pop()
        try:
            children = list(current.iterdir())
        except (PermissionError, OSError):
            continue
        for child in children:
            if child.is_symlink():
                continue
            if child.is_dir():
                if child.name in _SKIP_DIRS:
                    continue
                stack.append(child)
            elif child.is_file():
                yield child


def _excerpt(text: str, idx: int, needle_len: int, context: int = 60) -> str:
    start = max(0, idx - context)
    end = min(len(text), idx + needle_len + context)
    snippet = text[start:end].replace("\n", " ").replace("\r", " ").strip()
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(text) else ""
    return f"{prefix}{snippet}{suffix}"


def _display_rel(path: Path, home: Path) -> str:
    try:
        rel = path.resolve().relative_to(home)
    except (ValueError, OSError):
        return str(path)
    return f"~/{rel}" if str(rel) != "." else "~"


def _format_size(size: int) -> str:
    if size < 0:
        return "?"
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def _resolve_user_path(path: str) -> Path | None:
    """Resolve a user-supplied path relative to ``$HOME`` and confirm containment."""
    home = Path.home().resolve()
    raw_str = (path or "").strip()
    if raw_str in ("", "~"):
        return home
    raw = Path(raw_str).expanduser()
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
