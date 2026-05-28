"""Tests for packages/brain/src/june_brain/datadir."""

from __future__ import annotations

import json
import shutil

import june_brain.config
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _point_to(monkeypatch: pytest.MonkeyPatch, path: object) -> None:
    """Redirect june_brain.config.MEMORY_DIR to path (str)."""
    monkeypatch.setattr(june_brain.config, "MEMORY_DIR", str(path))


# ---------------------------------------------------------------------------
# Layout path resolution
# ---------------------------------------------------------------------------


class TestLayoutPaths:
    def test_all_paths_under_data_dir(self, tmp_path, monkeypatch):
        _point_to(monkeypatch, tmp_path)

        from june_brain.datadir import (
            character_dir,
            character_file,
            config_dir,
            data_dir,
            ledger_path,
            manifest_path,
            memory_dir,
            skills_dir,
            tasks_dir,
        )

        root = data_dir()
        assert root == tmp_path
        assert manifest_path().parent == root
        assert memory_dir().parent == root
        assert character_dir().parent == root
        assert character_file().parent == character_dir()
        assert skills_dir().parent == root
        assert tasks_dir().parent == root
        assert ledger_path().parent == tasks_dir()
        assert config_dir().parent == root

    def test_ensure_layout_creates_subdirs(self, tmp_path, monkeypatch):
        _point_to(monkeypatch, tmp_path)

        from june_brain.datadir import ensure_layout

        returned = ensure_layout()
        assert returned == tmp_path

        for subdir in ("memory", "character", "skills", "tasks", "config"):
            assert (tmp_path / subdir).is_dir(), f"missing subdir: {subdir}"

    def test_ensure_layout_idempotent(self, tmp_path, monkeypatch):
        _point_to(monkeypatch, tmp_path)

        from june_brain.datadir import ensure_layout

        ensure_layout()
        ensure_layout()  # must not raise


# ---------------------------------------------------------------------------
# init_data_dir + round-trip
# ---------------------------------------------------------------------------


class TestInitAndRoundTrip:
    def test_init_creates_manifest(self, tmp_path, monkeypatch):
        _point_to(monkeypatch, tmp_path)

        from june_brain.datadir import SCHEMA_VERSION, init_data_dir

        m = init_data_dir()
        assert m.schema_version == SCHEMA_VERSION
        assert (tmp_path / "manifest.json").exists()

    def test_round_trip_across_folder_move(self, tmp_path, monkeypatch):
        """Simulate 'move to a new machine': copy folder, re-point, reload."""
        _point_to(monkeypatch, tmp_path)

        from june_brain.datadir import SCHEMA_VERSION, init_data_dir, read_manifest

        original = init_data_dir()

        # Copy the whole data dir to a second location.
        new_dir = tmp_path.parent / "june_moved"
        shutil.copytree(tmp_path, new_dir)

        _point_to(monkeypatch, new_dir)
        reloaded = read_manifest()

        assert reloaded is not None
        assert reloaded.schema_version == original.schema_version
        assert reloaded.schema_version == SCHEMA_VERSION
        assert reloaded.created_at == original.created_at
        assert reloaded.june_version == original.june_version
        assert reloaded.contents == original.contents


# ---------------------------------------------------------------------------
# load_or_init
# ---------------------------------------------------------------------------


class TestLoadOrInit:
    def test_empty_dir_returns_fresh(self, tmp_path, monkeypatch):
        _point_to(monkeypatch, tmp_path)

        from june_brain.datadir import SCHEMA_VERSION, load_or_init

        manifest, fresh = load_or_init()
        assert fresh is True
        assert manifest.schema_version == SCHEMA_VERSION
        assert (tmp_path / "manifest.json").exists()
        for subdir in ("memory", "character", "skills", "tasks", "config"):
            assert (tmp_path / subdir).is_dir()

    def test_corrupt_manifest_recovers(self, tmp_path, monkeypatch):
        _point_to(monkeypatch, tmp_path)

        from june_brain.datadir import SCHEMA_VERSION, load_or_init

        # Write invalid JSON.
        (tmp_path / "manifest.json").write_text("{not json", encoding="utf-8")

        manifest, fresh = load_or_init()
        assert fresh is True
        assert manifest.schema_version == SCHEMA_VERSION

    def test_current_manifest_not_fresh(self, tmp_path, monkeypatch):
        _point_to(monkeypatch, tmp_path)

        from june_brain.datadir import init_data_dir, load_or_init

        init_data_dir()
        manifest, fresh = load_or_init()
        assert fresh is False

    def test_version_mismatch_migrates(self, tmp_path, monkeypatch):
        """An old-schema manifest is migrated to SCHEMA_VERSION without crashing."""
        _point_to(monkeypatch, tmp_path)

        from june_brain.datadir import (
            SCHEMA_VERSION,
            Manifest,
            ensure_layout,
            load_or_init,
            write_manifest,
        )

        # Write a manifest with a lower schema_version.
        old_version = SCHEMA_VERSION - 1  # may be 0
        ensure_layout()
        old_manifest = Manifest(
            schema_version=old_version,
            created_at="2020-01-01T00:00:00+00:00",
            june_version="0.0.1",
            contents=["memory/"],
        )
        write_manifest(old_manifest)

        manifest, fresh = load_or_init()
        assert fresh is False
        assert manifest.schema_version == SCHEMA_VERSION

        # Manifest on disk must also be bumped.
        from june_brain.datadir import read_manifest

        on_disk = read_manifest()
        assert on_disk is not None
        assert on_disk.schema_version == SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Manifest to_dict / from_dict
# ---------------------------------------------------------------------------


class TestManifestSerde:
    def test_round_trip_dict(self):
        from june_brain.datadir import Manifest

        m = Manifest(
            schema_version=1,
            created_at="2026-05-28T00:00:00+00:00",
            june_version="0.1.0",
            contents=["memory/", "character/"],
        )
        assert Manifest.from_dict(m.to_dict()) == m

    def test_json_file_is_readable(self, tmp_path, monkeypatch):
        _point_to(monkeypatch, tmp_path)

        from june_brain.datadir import init_data_dir

        init_data_dir()
        raw = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
        assert "schema_version" in raw
        assert "created_at" in raw
        assert "june_version" in raw
        assert "contents" in raw
