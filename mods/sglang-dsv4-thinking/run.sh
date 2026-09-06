#!/bin/bash
# run.sh — enable DSV4 thinking end to end (prompt + parser force)
#
# Bug: with no chat template, the dsv4 prompt ends each assistant turn with
# </think> (thinking disabled) unless SGLANG_DEFAULT_THINKING is set, and even
# with --reasoning-parser deepseek-v4 the parser only forces separation when
# the client sends chat_template_kwargs={"thinking": true} (opencode/Hermes
# never do). Net: the model never deliberates, thinking streams as plain
# output, clients never see reasoning blocks. vLLM serves the same weights
# with thinking enabled — most of the tool-call reliability gap.
#
# Fix (with recipe env SGLANG_DEFAULT_THINKING=1 + --reasoning-parser
# deepseek-v4 — NOTE the hyphen; the *tool* parser key is 'deepseekv4'):
# patch serving_chat._get_reasoning_from_request to honor the env default,
# so prompt mode, parser force, and grammar paths agree. See
# patch_dsv4_thinking.py.
set -euo pipefail

PREFIX="[dsv4-thinking]"
TARGET="/sgl-workspace/sglang/python/sglang/srt/entrypoints/openai/serving_chat.py"
SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/patch_dsv4_thinking.py"

echo "=== sglang DSV4 thinking-default mod ==="

if [ ! -f "$TARGET" ]; then
    echo "$PREFIX ERROR: serving_chat.py not found at $TARGET" >&2
    exit 1
fi

python3 "$SCRIPT"

# Clear stale bytecode so the patched module is recompiled.
find /sgl-workspace -path "*entrypoints/openai*" -name "__pycache__" -prune -exec rm -rf {} + 2>/dev/null || true

echo "=== OK: DSV4 thinking default installed ==="
