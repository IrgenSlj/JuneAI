"""CharacterBlock — June's persistent identity.

FixedTraits are IMMUTABLE by self-edit; honesty and the behavioral safety
floor live here.  LearnedTraits are editable; they shape tone and expression.
character_update() enforces the immutability guard at write time.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field, fields
from pathlib import Path

# ---------------------------------------------------------------------------
# Traits
# ---------------------------------------------------------------------------


@dataclass
class FixedTraits:
    candor: str = (
        "Tells the truth plainly and kindly; disagrees when it matters; never flatters."
    )
    care: str = (
        "Honest, never cruel; disagrees and declines kindly. "
        "Honesty and care are the same value."
    )
    not_a_professional: str = (
        "Not a therapist, doctor, lawyer, or financial advisor; "
        "gives information, helps the user think, and points to "
        "qualified humans for decisions."
    )
    wellbeing_over_engagement: str = (
        "Never optimizes for keeping the user talking; "
        "there is no metric that rewards more conversation."
    )
    privacy: str = (
        "Never volunteers heavy or sensitive memories unprompted; "
        "nothing leaves the device without visible consent."
    )

    def to_dict(self) -> dict[str, str]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> FixedTraits:
        known = {f.name: f.default for f in fields(cls)}  # type: ignore[misc]
        kwargs: dict[str, str] = {}
        for name, default in known.items():
            raw = data.get(name)
            kwargs[name] = str(raw) if raw is not None else str(default)
        return cls(**kwargs)


@dataclass
class LearnedTraits:
    tone: str = "warm, direct"
    humor_register: str = "light, occasional"
    warmth: str = "present, not effusive"
    verbosity: str = "concise by default"
    reads_user: list[str] = field(default_factory=list)
    interests: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> LearnedTraits:
        known = {f.name for f in fields(cls)}
        kwargs: dict[str, object] = {}
        for f in fields(cls):
            raw = data.get(f.name)
            if raw is None:
                kwargs[f.name] = f.default_factory() if f.default_factory is not None else f.default  # type: ignore[misc]
            elif f.name in ("reads_user", "interests"):
                kwargs[f.name] = list(raw) if isinstance(raw, list) else []
            else:
                kwargs[f.name] = str(raw)
        return cls(**{k: v for k, v in kwargs.items() if k in known})  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Immutability guard set
# ---------------------------------------------------------------------------

FIXED_FIELD_NAMES: frozenset[str] = frozenset(f.name for f in fields(FixedTraits))


# ---------------------------------------------------------------------------
# CharacterBlock
# ---------------------------------------------------------------------------


@dataclass
class CharacterBlock:
    fixed: FixedTraits
    learned: LearnedTraits
    version: int = 1

    def to_dict(self) -> dict[str, object]:
        return {
            "fixed": self.fixed.to_dict(),
            "learned": self.learned.to_dict(),
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> CharacterBlock:
        fixed_raw = data.get("fixed")
        learned_raw = data.get("learned")
        fixed = FixedTraits.from_dict(fixed_raw if isinstance(fixed_raw, dict) else {})
        learned = LearnedTraits.from_dict(learned_raw if isinstance(learned_raw, dict) else {})
        version = int(data["version"]) if "version" in data else 1  # type: ignore[call-overload]
        return cls(fixed=fixed, learned=learned, version=version)

    def to_block(self) -> str:
        """Render the always-in-context character description (section 2 of assembler)."""
        lines: list[str] = [
            "## June — character",
            "",
            "### Core (immutable)",
            f"Candor: {self.fixed.candor}",
            f"Care: {self.fixed.care}",
            f"Role boundary: {self.fixed.not_a_professional}",
            f"Engagement: {self.fixed.wellbeing_over_engagement}",
            f"Privacy: {self.fixed.privacy}",
            "",
            "### Expression",
            f"Tone: {self.learned.tone}",
            f"Humor: {self.learned.humor_register}",
            f"Warmth: {self.learned.warmth}",
            f"Verbosity: {self.learned.verbosity}",
        ]
        if self.learned.reads_user:
            lines.append("User observations: " + "; ".join(self.learned.reads_user))
        if self.learned.interests:
            lines.append("Shared interests: " + ", ".join(self.learned.interests))
        return "\n".join(lines)

    def save(self, path: Path | None = None) -> Path:
        """Atomic write to character_file() (or given path)."""
        from june_brain.datadir.layout import character_file, ensure_layout

        ensure_layout()
        target = path if path is not None else character_file()
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.to_dict(), indent=2, sort_keys=True)
        tmp_fd, tmp_name = tempfile.mkstemp(
            dir=target.parent, prefix=".persona_", suffix=".tmp"
        )
        try:
            os.write(tmp_fd, payload.encode("utf-8"))
            os.fsync(tmp_fd)
            os.close(tmp_fd)
            os.replace(tmp_name, target)
        except Exception:
            try:
                os.close(tmp_fd)
            except OSError:
                pass
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise
        return target

    @classmethod
    def load(cls, path: Path | None = None) -> CharacterBlock:
        """Read persona.json; if missing or corrupt return the seeded default."""
        from june_brain.datadir.layout import character_file

        target = path if path is not None else character_file()
        try:
            text = target.read_text(encoding="utf-8")
            data = json.loads(text)
            if not isinstance(data, dict):
                return _seeded_default()
            return cls.from_dict(data)
        except (OSError, json.JSONDecodeError, KeyError, ValueError, TypeError):
            return _seeded_default()


def _seeded_default() -> CharacterBlock:
    from june_brain.character.seed import seed_character

    return seed_character()


# ---------------------------------------------------------------------------
# character_update — Rung 1 self-edit with immutability guard
# ---------------------------------------------------------------------------


def character_update(
    updates: dict[str, object],
    *,
    path: Path | None = None,
    capability_ok: Callable[[], bool] | None = None,
) -> dict[str, object]:
    """Apply updates to LearnedTraits only.

    Returns a result dict:
      {"ok": True, "version": N}
      {"ok": False, "reason": "deepening_disabled"}
      {"ok": False, "reason": "immutable_fixed_traits", "rejected_keys": [...]}
    """
    if capability_ok is not None and not capability_ok():
        return {"ok": False, "reason": "deepening_disabled"}

    rejected = [k for k in updates if k in FIXED_FIELD_NAMES or k == "fixed"]
    if rejected:
        return {
            "ok": False,
            "reason": "immutable_fixed_traits",
            "rejected_keys": rejected,
        }

    block = CharacterBlock.load(path)

    learned_field_names = {f.name for f in fields(LearnedTraits)}
    for key, value in updates.items():
        if key not in learned_field_names:
            continue
        if key in ("reads_user", "interests"):
            setattr(block.learned, key, list(value) if isinstance(value, list) else [value])
        else:
            setattr(block.learned, key, str(value))

    block.version += 1
    block.save(path)
    return {"ok": True, "version": block.version}
