#!/bin/bash
# run.sh — tolerate OpenAI-function-shaped DSML invoke bodies (tool calling)
#
# Bug: the dsv4 TOOLS_TEMPLATE only demonstrates DSML XML parameter tags, but
# the detector also accepts direct-JSON invoke bodies without ever showing the
# model one. Once tool history exists in context, the model freelances
# Format-2 bodies in the OpenAI `function` shape it knows from pretraining:
#   <DSML>invoke name="bash">{"arguments": {"command": "..."}}</DSML>invoke>
# The detector passes the body through verbatim, so clients get
# arguments='{"arguments": {...}}' and fail schema validation. Reproduced 4/4
# on turn-2-style requests (0/3 on fresh turn-1).
#
# Fix: patch deepseekv32_detector.py (shared by the DeepSeekV4Detector
# subclass) to schema-aware unwrap function-shaped bodies to the inner
# arguments — unless the tool genuinely declares an "arguments" property —
# for complete bodies, plus stream suppression for partials still unfolding
# into the wrapper. Applies to BOTH direct-JSON bodies and XML params
# literally named "arguments" (the model copies that name from history).
# See patch_dsv4_fnshape.py.
# Companion: patch encoding_dsv4.py's history renderer so it never fabricates
# a param named "arguments" for unparseable args (that rendering teaches the
# model the wrapper shape). See patch_dsv4_history.py.
set -euo pipefail

PREFIX="[dsv4-fnshape]"
TARGET="/sgl-workspace/sglang/python/sglang/srt/function_call/deepseekv32_detector.py"
SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/patch_dsv4_fnshape.py"
TARGET2="/sgl-workspace/sglang/python/sglang/srt/entrypoints/openai/encoding_dsv4.py"
SCRIPT2="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/patch_dsv4_history.py"

echo "=== sglang DSV4 function-shape tolerance mod ==="

if [ ! -f "$TARGET" ]; then
    echo "$PREFIX ERROR: deepseekv32_detector.py not found at $TARGET" >&2
    exit 1
fi

python3 "$SCRIPT"

if [ ! -f "$TARGET2" ]; then
    echo "$PREFIX ERROR: encoding_dsv4.py not found at $TARGET2" >&2
    exit 1
fi

python3 "$SCRIPT2"

# Clear stale bytecode so the patched modules are recompiled.
find /sgl-workspace -path "*function_call*" -name "__pycache__" -prune -exec rm -rf {} + 2>/dev/null || true
find /sgl-workspace -path "*entrypoints/openai*" -name "__pycache__" -prune -exec rm -rf {} + 2>/dev/null || true

echo "=== OK: DSV4 function-shape tolerance installed ==="
