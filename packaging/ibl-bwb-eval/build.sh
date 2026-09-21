#!/bin/bash
# Builds the standalone `ibl-bwb-eval` sdist and wheel into dist/.
#
# Stages src/ibl_bwb_eval next to this directory's metadata under build/ and builds there,
# rather than pointing a build backend at ../../src: setuptools writes egg-info beside
# the sources it is given, which through a symlink lands in the repo's src/ tree.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
STAGE="$ROOT/build/ibl_bwb_eval-dist"

rm -rf "$STAGE" && mkdir -p "$STAGE/src"
cp -r "$ROOT/src/ibl_bwb_eval" "$STAGE/src/ibl_bwb_eval"
find "$STAGE/src" -name __pycache__ -type d -prune -exec rm -rf {} +
cp "$HERE/pyproject.toml" "$HERE/README.md" "$HERE/MANIFEST.in" "$HERE/CHANGELOG.md" "$ROOT/LICENSE" "$STAGE/"

cd "$STAGE"
uv build --out-dir "$ROOT/dist"
