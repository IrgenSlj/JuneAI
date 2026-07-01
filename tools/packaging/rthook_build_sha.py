"""PyInstaller runtime hook: bake JUNE_BUILD_SHA into the frozen sidecar.

There is no git checkout inside the packaged bundle, so
``june_brain.build_info.build_version()`` would otherwise degrade to
``"unknown"``. The packaging script (``tools/packaging/build-sidecar.sh``)
writes the short git SHA to ``_build_sha.txt`` and the spec bundles it at the
bundle root; this hook — which PyInstaller runs *before* the frozen program's
entry point imports ``june_api`` — reads that file and sets ``JUNE_BUILD_SHA``
in ``os.environ`` so ``GET /system`` reports the real build.

Everything here is best-effort and never fatal: if the file is absent (spec
built without the SHA step) or unreadable, boot proceeds and build_info
degrades gracefully. An explicit ``JUNE_BUILD_SHA`` already in the environment
(e.g. set by the desktop shell at spawn time) always wins.
"""

import os
import sys


def _inject_build_sha() -> None:
    if os.environ.get("JUNE_BUILD_SHA", "").strip():
        return  # An explicit value from the parent process wins.
    meipass = getattr(sys, "_MEIPASS", None)
    if not meipass:
        return
    sha_path = os.path.join(meipass, "_build_sha.txt")
    try:
        with open(sha_path, encoding="utf-8") as fh:
            sha = fh.read().strip()
    except OSError:
        return
    if sha:
        os.environ["JUNE_BUILD_SHA"] = sha


try:
    _inject_build_sha()
except Exception:  # noqa: BLE001 — never let the hook break the frozen boot.
    pass
