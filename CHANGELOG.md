# Changelog

## 1.0.4 - 2026-08-15

- Fixed `allow_overwrite=yes` on Windows for existing safetensors files.
- Moved the previous output to a temporary backup before saving, restore it if
  saving fails, and remove the backup after a successful save.
- Added a clear error when Windows has the destination file open or memory-mapped.

## 1.0.3 - 2026-08-15

- Added the Comfy Registry icon and icon metadata.

## 1.0.2 - 2026-08-15

- Added a working `allow_overwrite` switch to the Save LoRA node.
- Refuse to replace an existing output when the switch is `no`; overwrite it
  when the switch is `yes`.
- Enabled overwrite in the bundled example workflow for repeatable runs.

## 1.0.1 - 2026-08-15

- Restored the Show Text node and saved-path connection in the example workflow.
- Registered Krea2 Merge Save LoRA as a ComfyUI output node so it also runs
  without an attached display node.

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
