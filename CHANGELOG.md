# Changelog

## 1.3.0 - 2026-10-05

- Added the optional `svd_truncate` merge mode and a `target_rank` input. It
  composes LoRAs exactly, then limits each module to `target_rank` with an
  optimal SVD truncation, so the output rank does not grow with the number of
  inputs. Modules already within the budget stay exact, the result is
  deterministic, and the console reports the relative error of truncated
  modules. The default mode stays `exact_concat`.
- `target_rank` is an optional input, so workflows and API prompts saved before
  it existed load and run unchanged.
- Added tests that measure the truncation error against `exact_concat` and the
  dense SVD optimum, including convolution, both key styles, dtype and
  invalid-input checks.

## 1.2.1 - 2026-10-05

- Preserve missing LoRA filenames when loading workflows or changing folder
  filters, allowing ComfyUI to report the missing file instead of silently
  selecting another LoRA.
- Reject unsupported adapter data in both merge modes, including `diff`,
  `diff_b`, LoHa/LoKr tensors, normalization weights, and factor biases.
- Restrict Save LoRA to relative filenames and subfolders inside
  `krea2-merged-loras`. Reject absolute paths, parent traversal, symlink escapes,
  and unsupported output extensions before touching existing files.
- Use ComfyUI's safe checkpoint loader for Load and Apply, including `.sft`
  support. Save `.sft` files as safetensors as well.
- Keep alpha tensors in float32 in both merge modes, preventing BF16 from
  rounding an output rank of 259 to an alpha of 260.
- Warn when two inputs have no shared module names. Preserve their separate
  modules without attempting to translate PEFT/Kohya naming conventions.
- Add seeded random matrix and convolution checks, real checkpoint round trips,
  save-path and overwrite regression tests, and frontend lifecycle tests in CI.

## 1.2.0 - 2026-08-24

- Made `exact_concat` the default merge mode for mathematically exact weighted
  composition without cross terms.
- Kept `legacy_linear` available for existing workflows and added a runtime
  warning that it is an approximate compatibility mode.
- Added explicit `-4.0` to `4.0` ranges so negative weights can be entered in
  the ComfyUI frontend.
- Changed new Merge nodes to balanced `0.5` / `0.5` weights; values stored in
  existing workflows remain unchanged.
- Reject DoRA and LoCon `lora_mid` adapters instead of silently dropping or
  incorrectly scaling their auxiliary weights.
- Warn when a connected third or fourth LoRA has a zero weight.
- Reapply the saved folder filter after workflow configuration so filtered LoRA
  lists display correctly after loading a workflow.
- Updated the bundled workflow, tests, Registry metadata, and documentation.

## 1.1.0 - 2026-08-18

- Added the optional `exact_concat` merge mode for LoRAs with different ranks.
- Preserve `legacy_linear` as the default so existing workflows and outputs do
  not change.
- Apply exact weighted LoRA composition with output rank equal to the sum of
  participating ranks (for example, rank 4 + rank 16 becomes rank 20).
- Added validation for incomplete pairs, mixed key styles, non-rank dimension
  mismatches, and unsupported LoCon `lora_mid` tensors in exact mode.
- Added the new mode to the bundled workflow and regression tests.

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
