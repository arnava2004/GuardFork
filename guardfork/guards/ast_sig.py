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
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
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


def mutating_apps(sig: Signature) -> frozenset[str]:
    verbs = ("create", "update", "delete", "remove", "send", "post", "pay",
             "complete", "login", "logout", "add", "set", "write", "cancel")
    return frozenset(app for app, endpoint, _ in sig
                     if any(endpoint.lower().startswith(v) or v in endpoint.lower()
                            for v in verbs))
