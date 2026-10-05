# ComfyUI Krea2 Merge

<p align="center">
  <img src="assets/icon.png" alt="ComfyUI Krea2 Merge icon" width="200">
</p>

[![Tests](https://github.com/cyberdeliaAI/comfyui-krea2-merge/actions/workflows/tests.yml/badge.svg)](https://github.com/cyberdeliaAI/comfyui-krea2-merge/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

ComfyUI Krea2 Merge is a standalone set of nodes for exactly combining two to
four Krea 2 LoRAs without loading or modifying the base model. It supports the
PEFT/Diffusers `lora_A` + `lora_B` keys used by Krea 2 LoRAs, while retaining
support for Kohya `lora_down` + `lora_up` checkpoints.

This fork uses its own package name, extension name, node IDs, category, and
output folder. It can therefore be installed alongside the original
`comfyui-merge` extension without replacing its nodes.

## Features

- Merge two to four Krea 2 LoRAs without loading a base model.
- Support PEFT/Diffusers `lora_A` + `lora_B` state dictionaries.
- Retain Kohya `lora_down` + `lora_up` compatibility.
- Infer missing alpha values from the LoRA rank.
- Handle negative merge weights correctly for both `lora_B` and `lora_up`.
- Exactly merge LoRAs with equal or different ranks without unwanted cross terms.
- Optionally cap the output rank with `svd_truncate`, an optimal SVD truncation
  that reports its approximation error.
- Enter positive or negative weights from `-4.0` to `4.0`.
- Report unsupported files and incompatible tensor shapes clearly.
- Load `.safetensors`, `.sft`, and supported PyTorch checkpoints through ComfyUI's safe loader.
- Save only inside `ComfyUI/models/loras/krea2-merged-loras`, including subfolders.
- Keep alpha in float32 even when the factor tensors use FP16 or BF16.
- Keep merge results independent of input order for equal weights.

## Installation

1. Download or clone this repository into:

   ```text
   ComfyUI/custom_nodes/comfyui-krea2-merge
   ```

2. Restart ComfyUI.

The extension requires PyTorch (provided by ComfyUI). The `safetensors`
package is required for loading and saving `.safetensors` files and is included
in normal ComfyUI installations.

## Nodes

All nodes appear under **Krea2 Merge / LoRA**:

- **Krea2 Merge • Load LoRA** — loads a LoRA as a state dictionary.
- **Krea2 Merge • Merge LoRAs** — combines two to four LoRAs.
- **Krea2 Merge • Save LoRA** — saves the merged state dictionary.
- **Krea2 Merge • Apply LoRA** — loads and applies a LoRA to a connected model.

The internal node IDs are prefixed with `Krea2Merge_`, so workflows do not
collide with the original extension.

## Basic workflow

1. Add two **Krea2 Merge • Load LoRA** nodes and select your Krea 2 LoRAs.
2. Connect them to **Krea2 Merge • Merge LoRAs**.
3. Set `weight1` and `weight2`. Optional third and fourth inputs use `weight3`
   and `weight4`.
4. Keep the default `exact_concat` for a mathematically exact weighted merge.
   Choose `svd_truncate` instead when you want a smaller file: it limits every
   module to `target_rank` (see the compatibility notes). `legacy_linear`
   remains available only for compatibility with the original approximate
   factor-space behavior.
5. Select the output precision. `fp16` is the practical default; use `bf16` if
   that matches your Krea 2 setup.
6. Connect **Krea2 Merge • Save LoRA** and choose a filename ending in
   `.safetensors` or `.sft`, optionally inside a subfolder such as
   `characters/my_merge.safetensors`.
7. Set `allow_overwrite` to `yes` when repeated runs should replace the same
   output file. Keep it on `no` to protect an existing merge.

Since version 1.2.1, output paths must stay inside `krea2-merged-loras`.
Absolute paths, `..`, and symlinks leading outside that folder are rejected.
If an older workflow uses an absolute output path, replace it with a filename
or relative subfolder. Supported output extensions are `.safetensors`, `.sft`,
`.pt`, `.pth`, `.ckpt`, and `.bin`.

On Windows, overwrite first moves the previous output to a temporary backup. If
the new save fails, the original file is restored automatically. If Windows has
the destination open or memory-mapped, use another filename or restart ComfyUI.

An example is included at `workflow-examples/krea2-lora-merger.json`.

The example includes **Show Text** from
[ComfyUI-Custom-Scripts](https://github.com/pythongosssss/ComfyUI-Custom-Scripts)
to display the saved path. The Save node is also registered as an output node, so
saving still works when Show Text is removed from a custom workflow.

## Compatibility notes

- `exact_concat` is the default. It concatenates complete A/B or down/up pairs,
  avoiding the cross terms produced when factors are added separately. It also
  supports different ranks: a rank-4 plus rank-16 module becomes rank 20.
- Concatenation increases the output rank. Combining two rank-32 LoRAs produces
  rank 64; four produce rank 128. This increases file size and the temporary
  work needed while ComfyUI applies the LoRA.
- `svd_truncate` first composes the LoRAs exactly like `exact_concat`, then limits
  each module to at most `target_rank` (default 32, only used by this mode). It
  keeps the best possible rank-`target_rank` approximation of the merged weight
  delta (an optimal SVD truncation), so the output rank no longer grows with the
  number of inputs. Modules whose combined rank is already within `target_rank`
  stay exact. The result is deterministic, and the console reports the relative
  error of the truncated modules. The error is the share of the merged delta
  that lies beyond `target_rank`: it is small for LoRAs with a decaying
  spectrum and large for LoRAs whose rank is fully used, so pick a
  `target_rank` close to the summed input ranks when fidelity matters. Only
  linear and 1x1-up (LoCon-style) modules are supported; use `exact_concat`
  otherwise.
- `legacy_linear` preserves old saved workflows and factor-space results. It is
  approximate, introduces cross terms, and requires matching shapes and ranks.
- `exact_concat` and `svd_truncate` apply weights directly and ignore
  `force_same_strength`.
  For an even blend, start with `weight1 = 0.5` and `weight2 = 0.5`.
- All modes operate only on the LoRA files; none changes or saves the Krea 2
  base model.
- When a PEFT checkpoint has no embedded alpha tensors, the extension uses
  `alpha = rank` (unit scaling). This is a fallback assumption: the original
  training alpha cannot be recovered from the factors alone. Files that rely
  on a separate non-default training alpha need that scaling supplied first.
- Alpha is always saved in float32. `save_dtype` controls the factor tensors;
  it does not reduce alpha precision.
- DoRA, LoCon `lora_mid`, LoHa/LoKr, `diff`/`diff_b`, factor biases, and other
  unrecognized adapter keys are rejected in both modes instead of being
  silently omitted or merged incorrectly.
- Inputs should use the same base model and module naming convention. A warning
  appears when two inputs have no shared module names; separate target layers
  can be valid, but differently named aliases for the same layer are not
  converted or combined. Matching shapes alone do not establish compatibility.
- A missing LoRA filename in a saved workflow is retained so ComfyUI can report
  the missing file. Select its replacement explicitly in the Load or Apply node.
- A connected third or fourth LoRA with a zero weight is ignored with a clear
  console warning.

## Attribution

This project is derived from
[LingSss9/comfyui-merge](https://github.com/LingSss9/comfyui-merge) and retains
its MIT license. Krea 2 support, separate node IDs, validation, tests, and the
release branding were added in this fork.

Krea 2 and its model names belong to their respective owners. This community
extension is not an official Krea product.
