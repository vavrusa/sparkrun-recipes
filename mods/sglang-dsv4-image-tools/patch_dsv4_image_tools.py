#!/usr/bin/env python3
"""Patch sglang's encoding_dsv4.py to re-associate model-emitted image tokens
(IMAGE_PLACEHOLDER) with image content blocks, so a multimodal assistant
message can be re-encoded alongside tool calls.

Without this, `_validate_no_image_sp_tokens` / `_process_image_blocks` reject
message content carrying the placeholder outside a real image block, which
happens when the DSV4 tool-call path flattens an image to a placeholder text
token and the model echoes it back in its tool-call response (content,
thinking, or a normalized text block). Clients then echo that message back on
the next turn and encoding raises, e.g.:

- "Message content contains image special token ..." (string content)
- "Text block contains image placeholder ..." (list content, text block)
- "reasoning_content contains image special token ..." (thinking echo)

Re-association converts each placeholder back into an image content block
(using the images collected so far in the conversation, plus image blocks
already present in the same message), so validation passes and the image
survives. Placeholders that cannot resolve to any image (dangling token, or
reasoning_content which must stay a string) are stripped.
"""
import re
import sys

PATH = "/sgl-workspace/sglang/python/sglang/srt/entrypoints/openai/encoding_dsv4.py"

with open(PATH) as f:
    src = f.read()

orig = src

HELPER = '''def _split_dsv4_image_text(
    text: str, pool: List[Dict[str, Any]], start_idx: int
) -> Tuple[List[Any], int]:
    """Split text on IMAGE_PLACEHOLDER into text blocks + image record copies.

    Each placeholder maps to pool[(start_idx + k) % len(pool)]. Placeholders
    with an empty pool are dropped (dangling token, nothing to attach)."""
    parts = text.split(IMAGE_PLACEHOLDER)
    blocks: List[Any] = []
    idx = start_idx
    for i, part in enumerate(parts):
        if part:
            blocks.append({"type": "text", "text": part})
        if i < len(parts) - 1 and pool:
            blocks.append(dict(pool[idx % len(pool)]))
            idx += 1
    return blocks, idx


def _reassociate_dsv4_image_tokens(msg: Dict[str, Any], images: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Re-associate model-emitted image tokens (IMAGE_PLACEHOLDER) with image
    content blocks, so the message can be re-encoded alongside tool calls.
    Without this, _validate_no_image_sp_tokens / _process_image_blocks reject
    placeholder-carrying content. Image records pool = images collected from
    earlier messages plus image blocks already present in this same message."""
    content = msg.get("content")
    own: List[Dict[str, Any]] = []
    if isinstance(content, list):
        for block in content:
            if _is_image_block(block):
                try:
                    own.append(_extract_image(block))
                except ValueError:
                    pass
    pool = list(images) + own
    if isinstance(content, str) and IMAGE_PLACEHOLDER in content:
        if pool:
            msg["content"], _ = _split_dsv4_image_text(content, pool, 0)
        else:
            msg["content"] = content.replace(IMAGE_PLACEHOLDER, "")
    elif isinstance(content, list):
        new_blocks: List[Any] = []
        idx = 0
        changed = False
        for block in content:
            if (
                isinstance(block, dict)
                and block.get("type") == "text"
                and isinstance(block.get("text"), str)
                and IMAGE_PLACEHOLDER in block["text"]
            ):
                sub, idx = _split_dsv4_image_text(block["text"], pool, idx)
                new_blocks.extend(sub)
                changed = True
            else:
                new_blocks.append(block)
        if changed:
            msg["content"] = new_blocks
    # reasoning_content must stay a string (no blocks possible) and is
    # ephemeral thinking: strip any dangling placeholder.
    rc = msg.get("reasoning_content")
    if isinstance(rc, str) and IMAGE_PLACEHOLDER in rc:
        msg["reasoning_content"] = rc.replace(IMAGE_PLACEHOLDER, "")
    return msg


def _validate_no_image_sp_tokens(msg: Dict[str, Any]) -> None:'''

# Replace any older version of the helper (v1 had only the string-content
# branch) or insert fresh if absent. Idempotent: re-running is a no-op.
# NOTE: the regex CONSUMES the trailing `_validate...` def line (instead of a
# lookahead) because HELPER itself ends with that line — a lookahead would
# duplicate it on re-runs.
HELPER_RE = re.compile(
    r"(def _split_dsv4_image_text.*?)?def _reassociate_dsv4_image_tokens.*?"
    r"\ndef _validate_no_image_sp_tokens\(msg: Dict\[str, Any\]\) -> None:",
    re.DOTALL,
)
if "_reassociate_dsv4_image_tokens" in src:
    src = HELPER_RE.sub(HELPER, src, count=1)
elif "def _validate_no_image_sp_tokens(msg: Dict[str, Any]) -> None:" in src:
    src = src.replace(
        "def _validate_no_image_sp_tokens(msg: Dict[str, Any]) -> None:",
        HELPER,
        1,
    )
else:
    print(
        "[dsv4-image-tools] ERROR: _validate_no_image_sp_tokens anchor not found",
        file=sys.stderr,
    )
    sys.exit(1)

OLD_CALL = "        msg = copy.deepcopy(msg)\n        _validate_no_image_sp_tokens(msg)"
NEW_CALL = "        msg = copy.deepcopy(msg)\n        msg = _reassociate_dsv4_image_tokens(msg, images)\n        _validate_no_image_sp_tokens(msg)"
if NEW_CALL not in src:
    if OLD_CALL not in src:
        print(
            "[dsv4-image-tools] ERROR: process_image_messages call anchor not found",
            file=sys.stderr,
        )
        sys.exit(1)
    src = src.replace(OLD_CALL, NEW_CALL, 1)

if src == orig:
    print("[dsv4-image-tools] no changes needed (already patched)")
else:
    with open(PATH, "w") as f:
        f.write(src)
    print("[dsv4-image-tools] patched encoding_dsv4.py")
