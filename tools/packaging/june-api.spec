# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec — freeze `june-api` into a self-contained, relocatable
sidecar the Tauri desktop shell can spawn.

Build from a *fresh* venv that has `june-brain`, `june-api`, `openai`, and
`pyinstaller` installed — NEVER the project's `packages/brain/.venv` (the test
gate depends on it). See tools/packaging/README.md for the exact commands.

    pyinstaller --noconfirm \
        --distpath <out>/dist --workpath <out>/build \
        tools/packaging/june-api.spec

Produces an onedir bundle at <out>/dist/june-api/ whose `june-api` executable
boots uvicorn and serves the FastAPI app on JUNE_API_HOST:JUNE_API_PORT.
"""
import importlib.util

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_submodules,
)

# Entry point: june_api/__main__.py -> main() -> uvicorn.run("june_api.app:app").
# Resolved from the installed package so no launcher file is needed in the repo.
_entry = importlib.util.find_spec("june_api.__main__").origin

datas = []
binaries = []
hiddenimports = []

# june_brain + june_api: collect every submodule (uvicorn imports the app by
# string, routes import lazily) and every package-data file bundled with them
# (providers.toml, extractor_prompt.txt, registry_index.json) — all read at
# runtime via __file__ / importlib.resources, so they must ship as datas.
for pkg in ("june_brain", "june_api"):
    datas += collect_data_files(pkg)
    hiddenimports += collect_submodules(pkg)

# sqlite-vec ships vec0.dylib as a loadable SQLite extension. It is NOT linked
# to the Python package (it is loaded at runtime via conn.load_extension), so
# PyInstaller's dependency analysis misses it. Collect it as data so it lands
# at _internal/sqlite_vec/vec0.dylib, right next to sqlite_vec.__file__, which
# is where sqlite_vec.loadable_path() looks. THIS is the load-bearing line for
# semantic recall (ADR 0019).
datas += collect_data_files("sqlite_vec")

# uvicorn[standard] imports its loops/protocols/lifespan/logging submodules and
# the optional httptools/websockets/uvloop/watchfiles backends by string at
# runtime; collect_submodules alone can miss the string-imported leaves.
hiddenimports += collect_submodules("uvicorn")
hiddenimports += [
    "uvicorn.loops.auto",
    "uvicorn.loops.uvloop",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.http.httptools_impl",
    "uvicorn.protocols.http.h11_impl",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.protocols.websockets.websockets_impl",
    "uvicorn.lifespan.on",
    "uvicorn.logging",
    "httptools",
    "websockets",
    "uvloop",
    "watchfiles",
]

# openai is a real runtime dependency of june_brain (the Gemini provider uses
# the OpenAI-compatible client). NOTE: it is currently missing from
# packages/brain/pyproject.toml; it must be declared there and installed into
# the build venv, or the freeze silently drops the cloud provider.
hiddenimports += collect_submodules("openai")

a = Analysis(
    [_entry],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="june-api",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="june-api",
)
