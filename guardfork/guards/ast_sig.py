"""AST API-call signatures from AppWorld `execute` commands."""

from __future__ import annotations

import ast
import re

Signature = tuple[tuple[str, str, tuple[str, ...]], ...]
_EXEC_BLOCK = re.compile(
    r"appworld\s+exec(?:\s+--json)?(?:\s+<<['\"]?\w+['\"]?\n)?(.*)",
    re.DOTALL | re.IGNORECASE,
)


def python_from_command(command: str | None) -> str:
    if not command:
        return ""
    match = _EXEC_BLOCK.search(command)
    return (match.group(1) if match else command).strip()


def api_signature(command: str | None) -> Signature:
    """Ordered `(app, endpoint, sorted_kwarg_names)` tuples from `apis.app.endpoint(...)`."""
    source = python_from_command(command)
    if not source:
        return ()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ()
    out: list[tuple[str, str, tuple[str, ...]]] = []
    calls = sorted((n for n in ast.walk(tree) if isinstance(n, ast.Call)),
                   key=lambda n: (n.lineno, n.col_offset))
    for node in calls:
        func = node.func
        if not (isinstance(func, ast.Attribute) and isinstance(func.value, ast.Attribute)):
            continue
        if not (isinstance(func.value.value, ast.Name) and func.value.value.id == "apis"):
            continue
        app, endpoint = func.value.attr, func.attr
        names = sorted(
            kw.arg for kw in node.keywords if isinstance(kw.arg, str)
        )
        out.append((app, endpoint, tuple(names)))
    return tuple(out)


WRITE_VERBS = ("create", "update", "delete", "remove", "send", "post", "pay", "add", "set",
               "write", "cancel", "like", "follow", "approve", "deny", "move", "rename",
               "upload", "reply", "mark", "accept", "decline", "play", "pause", "start",
               "stop", "restore", "clear", "transfer", "request", "share")
READ_VERBS = ("show", "get", "list", "search", "read", "find", "status", "download")
NOT_STATE = ("login", "logout", "complete_task")


def is_write(endpoint: str) -> bool:
    e = endpoint.lower()
    if any(e.startswith(n) for n in NOT_STATE):
        return False
    return any(e.startswith(v) for v in WRITE_VERBS)


def is_read(endpoint: str) -> bool:
    return any(endpoint.lower().startswith(v) for v in READ_VERBS)


def mutating_apps(sig: Signature) -> frozenset[str]:
    return frozenset(app for app, endpoint, _ in sig if is_write(endpoint))
