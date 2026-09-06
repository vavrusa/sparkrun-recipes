#!/usr/bin/env python3
"""Patch sglang's encoding_dsv4.py history renderer so it never fabricates a
param named "arguments".

Root cause (companion to patch_dsv4_fnshape.py): `encode_arguments_to_dsml`
fell back to `{"arguments": <raw>}` for non-JSON arguments in echoed tool
history, rendering e.g. `<parameter name="arguments" ...>`. The model copies
that param name into its own calls (XML or JSON shape), and poisoned turns
keep teaching it — this is why a long session with wrapped calls in history
keeps failing intermittently. With the detector-side unwrap in place, the
renderer must stop manufacturing the shape: unparseable/non-dict arguments
now render as no params (fail-quiet), and non-dict JSON no longer risks an
AttributeError (.items() on a str) raising mid-encode.
"""
import sys

PATH = "/sgl-workspace/sglang/python/sglang/srt/entrypoints/openai/encoding_dsv4.py"

with open(PATH) as f:
    src = f.read()

orig = src

OLD = """    try:
        arguments = json.loads(tool_call["arguments"])
    except Exception as err:
        arguments = {"arguments": tool_call["arguments"]}"""
NEW = """    try:
        arguments = json.loads(tool_call.get("arguments", ""))
    except Exception:
        return ""
    if not isinstance(arguments, dict):
        # Never fabricate a param named "arguments": history must not teach
        # the model a wrapper shape the parsers then have to undo. Non-dict
        # JSON would also crash below on .items().
        return """""
if NEW not in src:
    if OLD not in src:
        print("[dsv4-fnshape] ERROR: history fallback anchor not found", file=sys.stderr)
        sys.exit(1)
    src = src.replace(OLD, NEW, 1)

if src == orig:
    print("[dsv4-fnshape] no changes needed (history already patched)")
else:
    with open(PATH, "w") as f:
        f.write(src)
    print("[dsv4-fnshape] patched encoding_dsv4 history renderer")
