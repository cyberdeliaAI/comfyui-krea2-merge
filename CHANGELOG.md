# Changelog

## 1.0.0 - 2026-08-15

- Rebranded the extension as ComfyUI Krea2 Merge.
- Added separate `Krea2Merge_` node IDs so the fork can coexist with the
  original `comfyui-merge` extension.
- Added Krea 2 PEFT/Diffusers `lora_A` and `lora_B` support.
- Added rank inference from A/down weights with B/up fallback.
- Added negative-weight handling for PEFT B matrices.
- Retained Kohya `lora_down` and `lora_up` compatibility.
- Replaced the previous missing-module `KeyError` with validation and safe
  fallbacks.
- Added clear errors for unsupported adapters and mismatched tensor shapes.
- Added a standalone example workflow and regression tests.
- Added Comfy Registry metadata for the `cyberdelia` publisher.
