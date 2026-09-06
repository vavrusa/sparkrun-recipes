#!/bin/bash
# run.sh — cap sglang DSV4 weight-loading threads (OOM workaround)
#
# sglang OOM-kills during weight load on the Spark: per-node weights (~76 GB,
# unified memory, cannot be swapped) plus the concurrent H2D weight-copy
# staging on the host exceed memory. Capping the weight-load ThreadPoolExecutor
# (the sglang DSV4 model loader) reduces concurrent staging buffers. This pairs
# with 64G of system swap (see the recipe header — required) so remaining host
# staging can swap instead of OOM-killing the process.
#
# Patches the editable sglang tree at
#   /sgl-workspace/sglang/python/sglang/srt/models/deepseek_v4.py
# replacing the bare `concurrent.futures.ThreadPoolExecutor()` with
# `concurrent.futures.ThreadPoolExecutor(max_workers=N)`.
set -euo pipefail

PREFIX="[sglang-weight-load-cap]"
WORKERS="${SGLANG_WEIGHT_LOAD_MAX_WORKERS:-4}"

echo "=== sglang weight-load thread cap mod ==="

MODEL_FILE=""
for cand in \
    /sgl-workspace/sglang/python/sglang/srt/models/deepseek_v4.py \
    "$(python3 -c 'import sglang,os;print(os.path.join(os.path.dirname(sglang.__file__),"srt","models","deepseek_v4.py"))' 2>/dev/null)"; do
    if [ -n "$cand" ] && [ -f "$cand" ]; then
        MODEL_FILE="$cand"
        break
    fi
done

if [ -z "$MODEL_FILE" ]; then
    echo "$PREFIX ERROR: could not locate deepseek_v4.py" >&2
    exit 1
fi
echo "$PREFIX model file: $MODEL_FILE"

python3 - "$MODEL_FILE" "$WORKERS" <<'PY'
import re, sys
path, workers = sys.argv[1], sys.argv[2]
with open(path) as f:
    src = f.read()

pat = re.compile(r"with concurrent\.futures\.ThreadPoolExecutor\([^)]*\) as executor:")
replacement = f"with concurrent.futures.ThreadPoolExecutor(max_workers={workers}) as executor:"

m = pat.search(src)
if m is None:
    print(f"[sglang-weight-load-cap] ERROR: ThreadPoolExecutor anchor not found", file=sys.stderr)
    sys.exit(1)

if m.group(0) == replacement:
    print(f"[sglang-weight-load-cap] already capped at max_workers={workers}; no change")
    sys.exit(0)

src = pat.sub(replacement, src, count=1)
with open(path, "w") as f:
    f.write(src)
print(f"[sglang-weight-load-cap] patched ThreadPoolExecutor -> max_workers={workers}")
PY

# Clear stale bytecode so the patched source is recompiled.
find /sgl-workspace -name "__pycache__" -path "*deepseek_v4*" -prune -exec rm -rf {} + 2>/dev/null || true

echo "=== OK: sglang weight-load threads capped at $WORKERS ==="
