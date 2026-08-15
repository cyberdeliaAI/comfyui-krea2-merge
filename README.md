# ComfyUI Krea2 Merge

ComfyUI Krea2 Merge is a standalone set of nodes for loading, combining, and
saving Krea 2 LoRAs. It supports the PEFT/Diffusers `lora_A` + `lora_B` keys
used by Krea 2 LoRAs, while retaining support for Kohya
`lora_down` + `lora_up` checkpoints.

This fork uses its own package name, extension name, node IDs, category, and
output folder. It can therefore be installed alongside the original
`comfyui-merge` extension without replacing its nodes.

## Features

- Merge two to four Krea 2 LoRAs without loading a base model.
- Support PEFT/Diffusers `lora_A` + `lora_B` state dictionaries.
- Retain Kohya `lora_down` + `lora_up` compatibility.
- Infer missing alpha values from the LoRA rank.
- Handle negative merge weights correctly for both `lora_B` and `lora_up`.
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
4. Select the output precision. `fp16` is the practical default; use `bf16` if
   that matches your Krea 2 setup.
5. Connect **Krea2 Merge • Save LoRA** and choose a filename ending in
   `.safetensors`.

An example is included at `workflow-examples/krea2-lora-merger.json`.

## Compatibility notes

- Input LoRAs must use matching tensor shapes for keys they share. In practice,
  this means the same Krea 2 base, target modules, and rank.
- When a PEFT checkpoint has no embedded alpha tensors, the extension uses
  `alpha = rank`. This is the only value that can be recovered from a standalone
  `.safetensors` state dictionary without its training configuration.
- DoRA magnitude vectors and other adapter methods are not merged.
- The merge algorithm intentionally follows the balanced, factor-space behavior
  of the original extension. It does not merge LoRAs into the Krea 2 base model.

## Attribution

This project is derived from
[LingSss9/comfyui-merge](https://github.com/LingSss9/comfyui-merge) and retains
its MIT license. Krea 2 support, separate node IDs, validation, tests, and the
release branding were added in this fork.

Krea 2 and its model names belong to their respective owners. This community
extension is not an official Krea product.
