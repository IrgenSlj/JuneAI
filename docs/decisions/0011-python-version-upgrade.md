# ADR 0011 — Upgrade Python Target to 3.13

## Status

Accepted

## Context

The project currently targets Python 3.10. This means:
- Cannot use `X | None` union syntax (PEP 604) — blocked by ruff rule UP007
- Cannot use `match` statements (PEP 636)
- Cannot use modern exception-handling features (`except*`, PEP 654)
- Cannot use `type` parameter syntax (PEP 695, Python 3.12+)
- Multiple typing improvements in 3.11+ (`Self` type, `NamedTuple` improvements, etc.)

The project is brand-new and has no compatibility constraints with older Python runtimes. The CI matrix tests 3.10/3.11/3.12, but we can simplify to a single modern target.

## Decision

- Bump minimum Python from 3.10 to 3.13 (latest stable release as of May 2026)
- Update `pyproject.toml` `requires-python` and `tool.mypy` `python_version`
- Remove ruff ignores for UP006, UP007, UP035, UP045
- Run `ruff check --fix --unsafe-fixes` to auto-migrate the codebase
- Update CI matrix to test only 3.13
- Perform a manual review of any remaining typing that ruff cannot auto-fix

## Consequences

**Positive:**
- Cleaner, more idiomatic code
- Better error messages from Python 3.11+ (with traceback improvements)
- Access to `Self` type (3.11), `type` parameter syntax (3.12), and improved `@dataclass` semantics
- Faster CPython (3.11 had significant performance improvements, 3.12/3.13 continued)
- Reduced CI matrix complexity (single Python version)

**Negative:**
- Users must have Python 3.13 installed (most modern systems already do; conda/uv handle this)
- Some library compatibility may lag (transient risk; all major dependencies support 3.13)
- One-time migration cost to fix typing annotations across ~50 source files

## Specific Migration Plan

1. Update `pyproject.toml`: `requires-python = ">=3.13"`, `tool.mypy.python_version = "3.13"`
2. Remove `lint.ignore = ["UP006", "UP007", "UP035", "UP045"]` from ruff config
3. Run `ruff check --fix --unsafe-fixes` to auto-convert annotations
4. Review remaining issues: especially `Optional[X]` → `X | None`, `Union[X, Y]` → `X | Y`, `Dict` → `dict`, `List` → `list`, `Tuple` → `tuple`, `Type` → `type`
5. Update CI workflow to test only `3.13`
6. Update `docs/setup/` to reference Python 3.13

## References

- PEP 604 — Allow writing union types as `X | Y`
- PEP 654 — Exception Groups and `except*`
- PEP 695 — Type Parameter Syntax
- CPython 3.13 release notes
