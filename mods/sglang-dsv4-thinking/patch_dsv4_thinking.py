#!/usr/bin/env python3
"""Patch sglang's serving_chat.py so the DSV4 thinking default actually takes
effect end to end.

Root cause: with no chat template, the dsv4 prompt ends each assistant turn
with </think> (thinking disabled) unless SGLANG_DEFAULT_THINKING is set, and
even with --reasoning-parser deepseek-v4 the parser only *forces* thinking
separation when the client sends chat_template_kwargs={"thinking": true}
(reasoning_default="explicit_thinking") — which opencode/Hermes never do.
Net effect: the model never deliberates, thinking (when emitted) streams as
plain output, and clients never see reasoning blocks. vLLM serves the same
weights with thinking enabled, which accounts for much of the tool-call
reliability gap between the stacks.

Fix (with recipe env SGLANG_DEFAULT_THINKING=1 + --reasoning-parser
deepseek-v4): make _get_reasoning_from_request return True for deepseek-v4
when the env default is on, so prompt mode (thinking suffix), parser force
(separation without client kwargs), and grammar paths all agree from one
source of truth. The v4 detector needs no <think> start tag handling beyond
what it has: force starts it in-reasoning and it splits at </think> (or at
the DSML tool marker).
"""
import sys

PATH = "/sgl-workspace/sglang/python/sglang/srt/entrypoints/openai/serving_chat.py"

with open(PATH) as f:
    src = f.read()

orig = src

OLD = """        if not self.reasoning_parser:
            return False

        if self.reasoning_parser == "minimax-m3":"""
NEW = """        if not self.reasoning_parser:
            return False

        # DSV4 thinking default: with SGLANG_DEFAULT_THINKING set (recipe),
        # thinking is on for every request without requiring clients to send
        # chat_template_kwargs={"thinking": True} (which they never do).
        # (envs is already imported module-wide in serving_chat.)
        try:
            if self.reasoning_parser == "deepseek-v4" and envs.SGLANG_DEFAULT_THINKING.get():
                return True
        except Exception:
            pass

        if self.reasoning_parser == "minimax-m3":"""
if NEW not in src:
    if OLD not in src:
        print("[dsv4-thinking] ERROR: reasoning anchor not found", file=sys.stderr)
        sys.exit(1)
    src = src.replace(OLD, NEW, 1)

if src == orig:
    print("[dsv4-thinking] no changes needed (already patched)")
else:
    with open(PATH, "w") as f:
        f.write(src)
    print("[dsv4-thinking] patched serving_chat thinking default")
