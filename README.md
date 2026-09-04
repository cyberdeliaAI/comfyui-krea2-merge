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
- Enter positive or negative weights from `-4.0` to `4.0`.
- Report unsupported files and incompatible tensor shapes clearly.
- Save to `ComfyUI/models/loras/krea2-merged-loras` by default.
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
   `legacy_linear` remains available only for compatibility with the original
   approximate factor-space behavior.
5. Select the output precision. `fp16` is the practical default; use `bf16` if
   that matches your Krea 2 setup.
6. Connect **Krea2 Merge • Save LoRA** and choose a filename ending in
   `.safetensors`.
7. Set `allow_overwrite` to `yes` when repeated runs should replace the same
   output file. Keep it on `no` to protect an existing merge.

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
- `legacy_linear` preserves old saved workflows and factor-space results. It is
  approximate, introduces cross terms, and requires matching shapes and ranks.
- `exact_concat` applies weights directly and ignores `force_same_strength`.
  For an even blend, start with `weight1 = 0.5` and `weight2 = 0.5`.
- Both modes operate only on the LoRA files; neither changes or saves the Krea 2
  base model.
- When a PEFT checkpoint has no embedded alpha tensors, the extension uses
  `alpha = rank`. This is the only value that can be recovered from a standalone
  `.safetensors` state dictionary without its training configuration.
- DoRA magnitude vectors, LoCon `lora_mid` tensors, and other adapter methods
  are rejected instead of being silently omitted or merged incorrectly.
- A connected third or fourth LoRA with a zero weight is ignored with a clear
  console warning.

## Attribution

This project is derived from
[LingSss9/comfyui-merge](https://github.com/LingSss9/comfyui-merge) and retains
its MIT license. Krea 2 support, separate node IDs, validation, tests, and the
release branding were added in this fork.

Krea 2 and its model names belong to their respective owners. This community
extension is not an official Krea product.
