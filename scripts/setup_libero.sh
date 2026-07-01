#!/usr/bin/env bash
# Clone LIBERO at the pinned commit, install it (legacy editable mode — see docs/macos.md),
# and seed ~/.libero/config.yaml so first import doesn't block on input().
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LIBERO_DIR="$REPO_ROOT/third_party/LIBERO"
LIBERO_COMMIT="8f1084e3132a39270c3a13ebe37270a43ece2a01"

if [ ! -d "$LIBERO_DIR" ]; then
    git clone https://github.com/Lifelong-Robot-Learning/LIBERO "$LIBERO_DIR"
fi
git -C "$LIBERO_DIR" fetch --depth 1 origin "$LIBERO_COMMIT" 2>/dev/null || true
git -C "$LIBERO_DIR" checkout "$LIBERO_COMMIT" 2>/dev/null || echo "warn: could not checkout pinned commit (shallow clone?); using HEAD"

pip install -e "$LIBERO_DIR" --config-settings editable_mode=compat --no-deps

LB="$LIBERO_DIR/libero/libero"
mkdir -p ~/.libero "$REPO_ROOT/data/libero_raw"
cat > ~/.libero/config.yaml <<EOF
assets: $LB/assets
bddl_files: $LB/bddl_files
benchmark_root: $LB
datasets: $REPO_ROOT/data/libero_raw
init_states: $LB/init_files
EOF
echo "LIBERO ready at $LIBERO_DIR (commit $LIBERO_COMMIT)"
