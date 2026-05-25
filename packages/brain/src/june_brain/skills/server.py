"""Minimal MCP stdio server helper shared by the skill packages.

Skills instantiate ``MCPStdioServer``, register tools with ``@server.tool``,
and call ``server.run()``. The helper handles:

- initialize / notifications/initialized handshake
- tools/list discovery
- tools/call dispatch with pydantic-validated arguments
- structured error returns (the agent sees a tool error, not a dead pipe)

Transport is newline-delimited JSON-RPC over stdin/stdout — compatible with
any MCP client (including the supervisor in :mod:`.supervisor`).
"""

from __future__ import annotations

import json
import logging
import sys
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

ToolFn = Callable[..., Any]


@dataclass
class _ToolEntry:
    name: str
    description: str
    input_schema: dict[str, Any]
    fn: ToolFn


class MCPStdioServer:
    """A tiny MCP-compliant server that speaks JSON-RPC 2.0 over stdio."""

    def __init__(self, name: str, version: str = "0.1.0") -> None:
        self.name = name
        self.version = version
        self._tools: dict[str, _ToolEntry] = {}

    def tool(
        self,
        name: str,
        description: str,
        input_schema: dict[str, Any] | None = None,
    ) -> Callable[[ToolFn], ToolFn]:
        """Register a tool handler.

        ``input_schema`` is the JSON Schema advertised to clients. If omitted,
        a permissive empty-object schema is used.
        """

        def decorator(fn: ToolFn) -> ToolFn:
            self._tools[name] = _ToolEntry(
                name=name,
                description=description,
                input_schema=input_schema or {"type": "object", "properties": {}},
                fn=fn,
            )
            return fn

        return decorator

    def run(self) -> None:
        """Block reading stdin and dispatching JSON-RPC messages until EOF."""
        # Unbuffered I/O: we want writes to flush after each message and reads
        # to return complete lines as they arrive from the supervisor.
        stdin = sys.stdin.buffer
        stdout = sys.stdout.buffer
        while True:
            raw = stdin.readline()
            if not raw:
                return
            try:
                message = json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError as exc:
                logger.warning("ignoring malformed stdin line: %s", exc)
                continue
            response = self._handle(message)
            if response is not None:
                stdout.write((json.dumps(response) + "\n").encode("utf-8"))
                stdout.flush()

    # ------------------------------------------------------------------ dispatch

    def _handle(self, message: dict[str, Any]) -> dict[str, Any] | None:
        method = message.get("method")
        message_id = message.get("id")
        params = message.get("params") or {}

        # Notifications have no id and never get a response.
        if message_id is None:
            return None

        try:
            if method == "initialize":
                return self._ok(
                    message_id,
                    {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": self.name, "version": self.version},
                    },
                )
            if method == "tools/list":
                return self._ok(
                    message_id,
                    {
                        "tools": [
                            {
                                "name": t.name,
                                "description": t.description,
                                "inputSchema": t.input_schema,
                            }
                            for t in self._tools.values()
                        ]
                    },
                )
            if method == "tools/call":
                return self._call_tool(message_id, params)
            return self._error(message_id, -32601, f"Method not found: {method}")
        except Exception as exc:  # noqa: BLE001
            logger.exception("handler error for method %s", method)
            return self._error(message_id, -32603, f"Internal error: {exc}")

    def _call_tool(self, message_id: Any, params: dict[str, Any]) -> dict[str, Any]:
        tool_name = params.get("name")
        arguments = params.get("arguments") or {}
        if not isinstance(tool_name, str) or tool_name not in self._tools:
            return self._error(message_id, -32602, f"Unknown tool: {tool_name!r}")
        if not isinstance(arguments, dict):
            return self._error(message_id, -32602, "arguments must be an object")
        entry = self._tools[tool_name]
        try:
            result = entry.fn(**arguments)
        except TypeError as exc:
            # wrong signature: report as a tool error, not a protocol error
            return self._ok(
                message_id,
                {
                    "content": [{"type": "text", "text": f"Argument error: {exc}"}],
                    "isError": True,
                },
            )
        except Exception as exc:  # noqa: BLE001
            return self._ok(
                message_id,
                {
                    "content": [
                        {"type": "text", "text": f"{exc}\n\n{traceback.format_exc()}"}
                    ],
                    "isError": True,
                },
            )
        return self._ok(message_id, _as_tool_result(result))

    def _ok(self, message_id: Any, result: Any) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": message_id, "result": result}

    def _error(self, message_id: Any, code: int, msg: str) -> dict[str, Any]:
        return {
            "jsonrpc": "2.0",
            "id": message_id,
            "error": {"code": code, "message": msg},
        }


def _as_tool_result(value: Any) -> dict[str, Any]:
    """Normalize a tool's return value to an MCP result envelope."""
    if isinstance(value, dict) and "content" in value:
        return value
    if isinstance(value, str):
        return {"content": [{"type": "text", "text": value}]}
    return {"content": [{"type": "text", "text": json.dumps(value, default=str)}]}
