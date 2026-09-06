#!/bin/bash
# run.sh — DSV4 official sampling defaults (temperature 1.0 / top_p 0.95)
#
# sglang has no --override-generation-config flag: with --sampling-defaults
# model it reads generation_config.json (temperature=1.0/top_p=1.0 here).
# This patches the resolved defaults to the official DeepSeek guidance,
# mirroring the eugr recipe's override-generation-config. Per-request values
# still win. See patch_dsv4_sampling.py.
set -euo pipefail

PREFIX="[dsv4-sampling]"
TARGET="/sgl-workspace/sglang/python/sglang/srt/configs/model_config.py"
SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/patch_dsv4_sampling.py"

echo "=== sglang DSV4 sampling-defaults mod ==="

if [ ! -f "$TARGET" ]; then
    echo "$PREFIX ERROR: model_config.py not found at $TARGET" >&2
    exit 1
fi

python3 "$SCRIPT"

# Clear stale bytecode so the patched module is recompiled.
find /sgl-workspace -path "*srt/configs*" -name "__pycache__" -prune -exec rm -rf {} + 2>/dev/null || true

echo "=== OK: DSV4 sampling defaults installed ==="
