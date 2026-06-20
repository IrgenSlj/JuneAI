"""June's tool abstraction — a tiny, dependency-free replacement for LangChain's
``StructuredTool`` / ``@tool``.

A tool is a plain Python callable exposed to the loop with a ``name``, a
``description`` (its docstring), an advertised argument schema (``args``), and an
``invoke(dict)`` entrypoint. The hand-written loop only ever uses
``.name`` / ``.description`` / ``.args`` / ``.invoke`` — this module reproduces
exactly that surface with no third-party dependency (CLAUDE.md invariant: no
dependency that can be implemented customly).

Some parameters are *injected* at dispatch time (e.g. the session ``state``
carrying the user_id) rather than supplied by the model. Mark them with
``Annotated[T, Inject]`` so they are excluded from the advertised ``args`` and
only filled when the dispatcher passes them to ``invoke``.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, Any, get_args, get_origin, get_type_hints


class _InjectMarker:
    """Sentinel placed in ``Annotated[...]`` metadata for injected params."""

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return "Inject"


Inject = _InjectMarker()


def _is_injected(annotation: Any) -> bool:
    """True when an annotation is ``Annotated[T, Inject]`` (or carries Inject)."""
    if get_origin(annotation) is not Annotated:
        return False
    return any(isinstance(m, _InjectMarker) for m in get_args(annotation)[1:])


def _type_name(annotation: Any) -> str:
    """A short, display-only type name for the advertised schema."""
    if get_origin(annotation) is Annotated:
        annotation = get_args(annotation)[0]
    return getattr(annotation, "__name__", str(annotation))


@dataclass
class Tool:
    """A callable tool with an advertised schema and an ``invoke`` entrypoint."""

    name: str
    description: str
    args: dict[str, dict[str, Any]]
    func: Callable[..., Any]
    injected: tuple[str, ...] = ()

    def invoke(self, arguments: dict[str, Any] | None = None) -> Any:
        """Call the underlying function with the supplied arguments.

        Only keys the function actually declares are forwarded — advertised
        args the model supplied, plus any injected params the dispatcher passed
        alongside them (e.g. ``{"mood": "good", "state": {"user_id": ...}}``).
        Anything omitted falls back to the function's own default.
        """
        arguments = dict(arguments or {})
        call_kwargs: dict[str, Any] = {}
        for pname in (*self.args.keys(), *self.injected):
            if pname in arguments:
                call_kwargs[pname] = arguments[pname]
        return self.func(**call_kwargs)


def tool(func: Callable[..., Any]) -> Tool:
    """Decorator turning a plain function into a :class:`Tool`.

    The advertised ``args`` are the function's parameters minus any marked
    ``Annotated[T, Inject]``. ``name`` is the function name; ``description`` is
    its docstring.
    """
    try:
        hints = get_type_hints(func, include_extras=True)
    except Exception:  # noqa: BLE001 — never let an unresolved hint break tool load
        hints = {}

    sig = inspect.signature(func)
    args_schema: dict[str, dict[str, Any]] = {}
    injected: list[str] = []

    for pname, param in sig.parameters.items():
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        annotation = hints.get(pname, param.annotation)
        if _is_injected(annotation):
            injected.append(pname)
            continue
        required = param.default is inspect.Parameter.empty
        args_schema[pname] = {
            "type": _type_name(annotation),
            "required": required,
            "default": None if required else param.default,
        }

    return Tool(
        name=func.__name__,
        description=inspect.getdoc(func) or "",
        args=args_schema,
        func=func,
        injected=tuple(injected),
    )
