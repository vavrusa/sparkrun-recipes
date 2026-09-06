#!/bin/bash
# run.sh — re-associate DSV4 image tokens with content blocks for tool calling
#
# Bug: enabling --tool-call-parser deepseekv4 selects sglang's "dsv4" chat
# encoding, which flattens image content blocks into the IMAGE_PLACEHOLDER
# token (<fullwidth-bar>deepseek_image<fullwidth-bar>). When the model echoes
# that token back in a tool-call response (content, thinking, or a normalized
# text block), the next turn's re-encode rejects it:
#  - "Message content contains image special token ..." (string content)
#  - "Text block contains image placeholder ..." (list content, text block)
#  - "reasoning_content contains image special token ..." (thinking echo)
# so tool calling with images fails.
#
# Fix: patch encoding_dsv4.py (_reassociate_dsv4_image_tokens, called from
# process_image_messages before validation) so placeholders are re-associated
# with image content blocks from the conversation's collected images (plus
# image blocks already in the same message). Unresolvable placeholders are
# stripped (dangling token; reasoning_content which must stay a string).
# The image thus survives alongside the tool call instead of 400ing.
#
# This complements the response-side --tool-call-parser deepseekv4; the fix
# cannot live in serving_chat.py because the response ChatMessage.content is a
# str and cannot hold content blocks.
set -euo pipefail

PREFIX="[dsv4-image-tools]"
TARGET="/sgl-workspace/sglang/python/sglang/srt/entrypoints/openai/encoding_dsv4.py"
SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/patch_dsv4_image_tools.py"

echo "=== sglang DSV4 image-tools re-association mod ==="

if [ ! -f "$TARGET" ]; then
    echo "$PREFIX ERROR: encoding_dsv4.py not found at $TARGET" >&2
    exit 1
fi

python3 "$SCRIPT"

# Clear stale bytecode so the patched module is recompiled.
find /sgl-workspace -path "*entrypoints/openai*" -name "__pycache__" -prune -exec rm -rf {} + 2>/dev/null || true

echo "=== OK: DSV4 image-tool re-association installed ==="
