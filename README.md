# sparkrun-recipes

Shareable [sparkrun](https://github.com/eugr/sparkrun) recipes and mods for
serving models on DGX Spark clusters.

## DeepSeek-V4-Flash-Vision-Exp (eugr b12x)

`deepseek-v4-flash-vision-exp-eugr.yaml` serves
`deepseek-ai/DeepSeek-V4-Flash-Vision-Exp` — DeepSeek's first multimodal V4 —
with **native image input** and DSpark speculative decoding intact. It is built
on the eugr b12x runtime image and pulls in two mods:

- `mods/deepseek-v4-vision-exp` — injects the 32-block ViT + aligner + image
  embeddings and the multimodal processor into vLLM's vendored
  `deepseek_v4/nvidia/model.py` (see its README).
- `mods/instanttensor-hybrid-draft-loader` — keeps InstantTensor for the target
  model while allowing the embedded DSpark draft to load via lazy safetensors.

### Layout

```
deepseek-v4-flash-vision-exp-eugr.yaml
mods/
  deepseek-v4-vision-exp/          # native vision mod (recipe's vision support)
  instanttensor-hybrid-draft-loader/
```

### Quick start

```bash
sparkrun run deepseek-v4-flash-vision-exp-eugr.yaml --cluster <name>
```

The recipe is tuned for TP=2 on a 2-node cluster. Adjust `tensor_parallel`,
`min_nodes`, and `gpu_memory_utilization` for your topology. See the mod
READMEs for the Vision-Exp-specific constraints (k must be a multiple of 3,
images in user messages only, the b12x `rms_norm_eps` fix).

## DeepSeek-V4-Flash-Vision-Exp (SGLang B12X)

`deepseek-v4-flash-vision-exp-sglang-b12x.yaml` serves the same checkpoint on
SGLang (DGX-Spark-only preview image `lmsysorg/sglang:dev-v4f-2dgx-v2`) with
DSPARK speculative decoding. B12X + DSV4 support is baked into the image and
enabled via env vars; five small mods patch the serving layer at launch:

- `mods/sglang-weight-load-cap` — caps the DSV4 weight-load thread pool so
  host staging stays manageable (pairs with swap, below).
- `mods/sglang-dsv4-image-tools` — re-associates model-echoed image tokens
  with image content blocks so tool calling works with images.
- `mods/sglang-dsv4-fnshape` — tolerates OpenAI-function-shaped tool-call
  bodies (`{"arguments": …}`) the model emits once tool history exists, and
  flushes unclosed invokes at stream end instead of dropping their arguments.
- `mods/sglang-dsv4-thinking` — prompts for thinking by default and forces
  reasoning separation, so clients get `reasoning_content` blocks.
- `mods/sglang-dsv4-sampling` — official sampling defaults
  (temperature=1.0/top_p=0.95); sglang has no override-generation-config flag.

Sampler/effort defaults mirror the vLLM side: temperature=1.0, top_p=0.95,
reasoning effort high. Note the spellings: `--tool-call-parser deepseekv4`
(no hyphen) vs `--reasoning-parser deepseek-v4` (hyphen) — two registries.

### Requirements

- 2 DGX Spark nodes (TP=2), tested at context 393216 with
  `gpu_memory_utilization` 0.85.
- **64GB of system swap on each node** (e.g. `fallocate -l 64G /swapfile &&
  chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile`). The FP8→FP4
  shared-expert conversion at load stages through host memory and OOMs
  without it, even with the thread-cap mod.

### Quick start

```bash
sparkrun run deepseek-v4-flash-vision-exp-sglang-b12x.yaml --cluster <name>
```
