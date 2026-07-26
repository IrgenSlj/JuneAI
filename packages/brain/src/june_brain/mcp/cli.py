"""``june-mcp`` — run the memory server, and manage who may read it.

Consent needs a command line, not only a UI. The person granting access is
sitting in Claude Desktop or Cursor when the denial appears, June's desktop app
may not be running at all, and the MCP server is spawned by the client rather
than by June. A grant flow that requires the app to be open is a grant flow that
fails exactly when it is needed.

    june-mcp serve                          # what the MCP client launches
    june-mcp grant claude-desktop search_memory
    june-mcp list
    june-mcp revoke claude-desktop          # all tools for that client
"""

from __future__ import annotations

import argparse
import sys
import time

from .consent import GRANTABLE_TOOLS, ConsentStore, grants_path


def _fmt_age(seconds: float) -> str:
    if seconds < 90:
        return f"{int(seconds)}s ago"
    if seconds < 5400:
        return f"{int(seconds // 60)}m ago"
    if seconds < 172800:
        return f"{int(seconds // 3600)}h ago"
    return f"{int(seconds // 86400)}d ago"


def cmd_serve(_args: argparse.Namespace) -> int:
    from .server import build_server

    build_server().run()
    return 0


def cmd_grant(args: argparse.Namespace) -> int:
    store = ConsentStore()
    tools = GRANTABLE_TOOLS if args.tool == "all" else [args.tool]
    for tool in tools:
        if tool not in GRANTABLE_TOOLS:
            print(
                f"'{tool}' is not a grantable tool. "
                f"Choose from: {', '.join(GRANTABLE_TOOLS)}, or 'all'.",
                file=sys.stderr,
            )
            return 2
    for tool in tools:
        store.grant(args.client, tool)
        print(f"granted {args.client} -> {tool}")
    print(f"\nStored in {grants_path()}. Revoke with: june-mcp revoke {args.client}")
    return 0


def cmd_revoke(args: argparse.Namespace) -> int:
    store = ConsentStore()
    removed = store.revoke(args.client, args.tool)
    if not removed:
        print(f"nothing to revoke for {args.client}", file=sys.stderr)
        return 1
    scope = args.tool or "all tools"
    print(f"revoked {args.client} -> {scope} ({removed} grant(s))")
    print("Takes effect on the client's very next call.")
    return 0


def cmd_list(_args: argparse.Namespace) -> int:
    grants = ConsentStore().list_grants()
    if not grants:
        print("No client may read June's memory.")
        print(f"(grants file: {grants_path()})")
        return 0
    now = time.time()
    print(f"{'CLIENT':<24} {'TOOL':<16} {'USES':>5}  LAST USED")
    for g in sorted(grants, key=lambda x: (x.client, x.tool)):
        last = _fmt_age(now - g.last_used) if g.last_used else "never"
        print(f"{g.client:<24} {g.tool:<16} {g.uses:>5}  {last}")
    print(f"\n{len(grants)} grant(s) in {grants_path()}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="june-mcp",
        description="June's MCP memory server and its consent grants.",
    )
    subs = parser.add_subparsers(dest="command")

    p_serve = subs.add_parser("serve", help="run the MCP server on stdio")
    p_serve.set_defaults(func=cmd_serve)

    p_grant = subs.add_parser("grant", help="allow a client to call a tool")
    p_grant.add_argument("client", help="client id, e.g. claude-desktop")
    p_grant.add_argument(
        "tool",
        nargs="?",
        default="all",
        help=f"one of {', '.join(GRANTABLE_TOOLS)}, or 'all' (default)",
    )
    p_grant.set_defaults(func=cmd_grant)

    p_revoke = subs.add_parser("revoke", help="withdraw access")
    p_revoke.add_argument("client")
    p_revoke.add_argument("tool", nargs="?", default=None, help="omit to revoke all")
    p_revoke.set_defaults(func=cmd_revoke)

    p_list = subs.add_parser("list", help="show live grants")
    p_list.set_defaults(func=cmd_list)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    # Bare `june-mcp` serves, because that is what an MCP client config invokes
    # and clients do not pass subcommands.
    if not getattr(args, "command", None):
        return cmd_serve(args)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
