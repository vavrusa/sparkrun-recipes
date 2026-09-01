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
