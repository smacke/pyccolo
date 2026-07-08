#!/usr/bin/env bash
#
# Build the pyccolo JupyterLite demo site (the target of the "launch in
# JupyterLite" README badge) into ./dist.
#
# It builds everything from *this checkout* rather than PyPI so the demo always
# reflects the current source:
#   1. the pyccolo wheel (runs in the browser via piplite),
#   2. the pure-Python runtime dependency wheels, bundled offline for a fast,
#      robust load (no CDN/PyPI hit at runtime).
#
# Usage:  bash jupyterlite/build.sh [OUTPUT_DIR]   # default: ./dist
# Prereqs: Python 3.11+ and network access to build.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
DIST="${1:-$ROOT/dist}"
# `jupyter lite build` auto-discovers (copies + indexes into pypi/all.json) any
# wheel placed in <lite-dir>/pypi -- so dropping our wheels here is all that's
# needed to make `%pip install pyccolo` resolve them offline via piplite.
WHEELS="$HERE/pypi"

rm -rf "$WHEELS" "$DIST"
mkdir -p "$WHEELS"

# Make sure `pip` and the `build` frontend are importable in the active
# interpreter, whether it's a uv venv (which ships without pip), a plain venv,
# or a system Python. The rest of the script then stays on the standard
# `python -m pip` / `python -m build` so it behaves identically everywhere --
# in particular `pip download` (below) has no `uv pip` equivalent.
echo "==> Ensuring pip + build are available"
python -m pip --version >/dev/null 2>&1 \
  || python -m ensurepip --upgrade >/dev/null 2>&1 \
  || { command -v uv >/dev/null 2>&1 && uv pip install pip; }
python -c "import build" >/dev/null 2>&1 || python -m pip install --quiet build

echo "==> Building the pyccolo wheel from this checkout"
python -m build --wheel "$ROOT" --outdir "$WHEELS"

echo "==> Bundling pure-Python runtime deps offline (fast, robust load)"
# pyccolo's runtime closure is traitlets + typing_extensions (both pure Python).
# `comm` is NOT a pyccolo dependency, but the Pyodide kernel's Jupyter stack
# pulls it in, and micropip's environment-consistency resolution during
# `%pip install pyccolo` fails outright if `comm` can't be resolved from the
# bundled index -- so bundle it (and its dep traitlets) too. traitlets/
# typing_extensions ship in the Pyodide env already; bundling them is harmless
# insurance so the demo still loads if that ever changes.
python -m pip download --only-binary=:all: \
  --implementation py --abi none --platform any \
  comm traitlets typing-extensions -d "$WHEELS"
# keep only universal (pure-Python) wheels
find "$WHEELS" -name '*.whl' ! -name '*-none-any.whl' -delete || true
echo "  bundled $(ls "$WHEELS"/*.whl | wc -l | tr -d ' ') wheels into $(basename "$WHEELS")/"

echo "==> Building the JupyterLite site"
python -m pip install jupyterlite-core jupyterlite-pyodide-kernel jupyter-server
jupyter lite build --contents "$HERE/content" --lite-dir "$HERE" --output-dir "$DIST"

echo "==> Done. Serve locally with:  python -m http.server -d '$DIST' 8000"
echo "    then open http://127.0.0.1:8000/lab/index.html?path=demo.ipynb"
