#!/usr/bin/env bash
# Build (or serve) the swarmstate documentation site.
#
#   ./scripts/build-docs.sh            # build into site/, warnings are errors
#   ./scripts/build-docs.sh --serve    # live-reloading server on 127.0.0.1:8000
#
# The API reference is generated from the installed package, so swarmstate has to
# be importable: run `maturin develop --release` (or `pip install -e .`) first.
set -euo pipefail

cd "$(dirname "$0")/.."

if ! python -c "import mkdocs" >/dev/null 2>&1; then
    echo "mkdocs is not installed. Run:  pip install -e \".[docs]\"" >&2
    exit 1
fi

if ! python -c "import swarmstate" >/dev/null 2>&1; then
    echo "swarmstate is not importable, so the API reference would be empty." >&2
    echo "Run:  maturin develop --release" >&2
    exit 1
fi

case "${1:-}" in
    --serve)
        shift
        exec python -m mkdocs serve --strict "$@"
        ;;
    "")
        exec python -m mkdocs build --strict
        ;;
    *)
        exec python -m mkdocs "$@"
        ;;
esac
