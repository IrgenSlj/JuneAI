#!/usr/bin/env bash
#
# build-sidecar.sh — freeze `june-api` with PyInstaller and STAGE the onedir
# bundle where Tauri picks it up as a resource that lands at
# `<App Resources>/june-api/`.
#
# Pairs with:
#   - tools/packaging/june-api.spec        (the frozen bundle definition)
#   - apps/desktop/src-tauri/tauri.conf.json  (`bundle.resources` maps the
#       staged dir -> Resources/june-api/)
#   - apps/desktop/src-tauri/src/sidecar.rs (resolves `june-api/june-api` under
#       BaseDirectory::Resource and spawns it)
#
# Invariants:
#   * The build venv and all scratch build dirs live OUTSIDE the repo. The
#     project venv (packages/brain/.venv) is never touched — the test gate
#     depends on it, and PyInstaller pulls in build-only deps.
#   * The staged onedir (apps/desktop/src-tauri/binaries/june-api/) is a build
#     artifact and is gitignored — never commit the ~40MB bundle.
#   * Idempotent / re-runnable: the staging dir is cleaned first every run.
#
# Overridable via env:
#   JUNE_SIDECAR_BUILD_ROOT   where the venv + scratch dirs live (default: a
#                             stable dir under $TMPDIR, always outside the repo)
#   JUNE_SIDECAR_FRESH_VENV=1 recreate the build venv from scratch
#   JUNE_BUILD_PYTHON         python interpreter for the venv (default: python3)
set -euo pipefail

log() { printf '\n[build-sidecar] %s\n' "$*" >&2; }
die() { printf '\n[build-sidecar] ERROR: %s\n' "$*" >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SPEC="${SCRIPT_DIR}/june-api.spec"
[[ -f "${SPEC}" ]] || die "spec not found at ${SPEC}"

# --- Build location: always OUTSIDE the repo -------------------------------
BUILD_ROOT="${JUNE_SIDECAR_BUILD_ROOT:-${TMPDIR:-/tmp}/june-sidecar-build}"
BUILD_ROOT="${BUILD_ROOT%/}"
# Refuse to place build scratch inside the repo — that would risk committing it
# and could shadow the project venv.
case "${BUILD_ROOT}/" in
  "${REPO_ROOT}/"*) die "JUNE_SIDECAR_BUILD_ROOT (${BUILD_ROOT}) must be OUTSIDE the repo" ;;
esac

BUILD_VENV="${BUILD_ROOT}/build-venv"
DIST_DIR="${BUILD_ROOT}/dist"
WORK_DIR="${BUILD_ROOT}/work"
SHA_FILE="${BUILD_ROOT}/_build_sha.txt"

# --- Staging location: INSIDE the repo (gitignored) ------------------------
STAGE_PARENT="${REPO_ROOT}/apps/desktop/src-tauri/binaries"
STAGE_DIR="${STAGE_PARENT}/june-api"

PYTHON_BIN="${JUNE_BUILD_PYTHON:-python3}"

log "repo root : ${REPO_ROOT}"
log "build root: ${BUILD_ROOT} (outside repo)"
log "stage dir : ${STAGE_DIR}"

mkdir -p "${BUILD_ROOT}"

# --- 1. Build venv (reused unless FRESH requested or missing) --------------
if [[ "${JUNE_SIDECAR_FRESH_VENV:-0}" == "1" && -d "${BUILD_VENV}" ]]; then
  log "removing existing build venv (JUNE_SIDECAR_FRESH_VENV=1)"
  rm -rf "${BUILD_VENV}"
fi
if [[ ! -x "${BUILD_VENV}/bin/python" ]]; then
  log "creating build venv at ${BUILD_VENV}"
  "${PYTHON_BIN}" -m venv "${BUILD_VENV}"
fi
VENV_PY="${BUILD_VENV}/bin/python"

log "installing brain + api + first-party skills + pyinstaller into build venv"
"${VENV_PY}" -m pip install --quiet --upgrade pip
# First-party skills to bundle: the enabled set in DEFAULT_MANIFEST. Each lives
# at skills/<name>/ with a pyproject naming june_skill_<name> and depends on
# june-brain. telegram is intentionally excluded — it is disabled by default and
# needs an optional extra (python-telegram-bot).
FIRST_PARTY_SKILLS=(calendar health research files daily)
SKILL_PATHS=()
for _skill in "${FIRST_PARTY_SKILLS[@]}"; do
  _skill_dir="${REPO_ROOT}/skills/${_skill}"
  [[ -f "${_skill_dir}/pyproject.toml" ]] || die "first-party skill not found: ${_skill_dir}"
  SKILL_PATHS+=("${_skill_dir}")
done
# One install command so pip resolves each skill's and the api's `june-brain`
# dependency against the local brain package installed in the same resolution.
# `openai` is a declared brain dependency, so it (and pydantic/httpx/etc.) comes
# along automatically; the research/files skills pull their own extras (httpx,
# pypdf, trafilatura) from PyPI.
"${VENV_PY}" -m pip install --quiet \
  "${REPO_ROOT}/packages/brain" \
  "${REPO_ROOT}/packages/api" \
  "${SKILL_PATHS[@]}" \
  pyinstaller

# --- 2. Build-SHA stamp (best-effort; see rthook_build_sha.py) -------------
if SHA="$(git -C "${REPO_ROOT}" rev-parse --short HEAD 2>/dev/null)" && [[ -n "${SHA}" ]]; then
  printf '%s' "${SHA}" > "${SHA_FILE}"
  export JUNE_BUILD_SHA_FILE="${SHA_FILE}"
  log "build SHA ${SHA} -> ${SHA_FILE} (bundled via spec + runtime hook)"
else
  rm -f "${SHA_FILE}"
  unset JUNE_BUILD_SHA_FILE || true
  log "git SHA unavailable; /system will report version 'unknown' (non-fatal)"
fi

# --- 3. Freeze --------------------------------------------------------------
rm -rf "${DIST_DIR}" "${WORK_DIR}"
log "freezing june-api with PyInstaller"
"${BUILD_VENV}/bin/pyinstaller" --noconfirm --clean \
  --distpath "${DIST_DIR}" \
  --workpath "${WORK_DIR}" \
  "${SPEC}"

FROZEN="${DIST_DIR}/june-api"
[[ -x "${FROZEN}/june-api" ]] || die "expected frozen executable at ${FROZEN}/june-api"

# --- 4. Stage into the repo (idempotent) -----------------------------------
log "staging onedir -> ${STAGE_DIR}"
rm -rf "${STAGE_DIR}"
mkdir -p "${STAGE_PARENT}"
cp -R "${FROZEN}" "${STAGE_DIR}"

# Sanity: Tauri (bundle.resources "binaries/june-api/") + sidecar.rs
# (resolve "june-api/june-api") require the executable exactly here.
[[ -x "${STAGE_DIR}/june-api" ]] || die "staged executable missing at ${STAGE_DIR}/june-api"

log "done. staged executable: ${STAGE_DIR}/june-api"
log "bundles into: <App Resources>/june-api/june-api"
