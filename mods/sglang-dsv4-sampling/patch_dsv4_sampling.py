#!/usr/bin/env python3
"""Patch sglang's model_config.py to default DSV4 sampling to the official
DeepSeek guidance, mirroring the eugr vLLM recipe's
--override-generation-config '{"temperature":1.0,"top_p":0.95}'.

sglang has no override-generation-config flag: with --sampling-defaults model
(the default) it reads generation_config.json, which ships temperature=1.0 /
top_p=1.0 for this checkpoint. This patch forces the resolved *defaults* to
temperature=1.0 / top_p=0.95. Explicit per-request values still win (the dict
is only consulted as fallback), and temperature is additionally enforced on
the proxy chain by force_temp_hook.
"""
import sys

PATH = "/sgl-workspace/sglang/python/sglang/srt/configs/model_config.py"

with open(PATH) as f:
    src = f.read()

orig = src

OLD = """        default_sampling_params = {
            p: config.get(p) for p in available_params if config.get(p) is not None
        }

        return default_sampling_params"""
NEW = """        default_sampling_params = {
            p: config.get(p) for p in available_params if config.get(p) is not None
        }

        # DSV4 official sampling guidance (mirrors eugr
        # --override-generation-config). Defaults only: explicit per-request
        # values still take precedence downstream.
        default_sampling_params["temperature"] = 1.0
        default_sampling_params["top_p"] = 0.95

        return default_sampling_params"""
if NEW not in src:
    if OLD not in src:
        print("[dsv4-sampling] ERROR: defaults anchor not found", file=sys.stderr)
        sys.exit(1)
    src = src.replace(OLD, NEW, 1)

if src == orig:
    print("[dsv4-sampling] no changes needed (already patched)")
else:
    with open(PATH, "w") as f:
        f.write(src)
    print("[dsv4-sampling] patched model_config sampling defaults")
