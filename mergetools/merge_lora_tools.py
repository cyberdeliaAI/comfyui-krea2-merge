import torch
import os
import ntpath
import math
import uuid
import folder_paths

# -------------------------------------------------------------------
# Optional safetensors support flag (keep both names for legacy checks)
# -------------------------------------------------------------------
try:
    from safetensors.torch import save_file as safe_save
    SAFETENSORS = True
    safetensors_available = True   # legacy alias
except ImportError:
    SAFETENSORS = False
    safetensors_available = False
    safe_save = None

# -------------------------------------------------------------------
# Build default output directory: <ComfyUI>/models/loras/krea2-merged-loras
# -------------------------------------------------------------------
lora_base_path = folder_paths.get_folder_paths("loras")[0]
OUTPUT_DIR = os.path.join(lora_base_path, "krea2-merged-loras")


LORA_DOWN_MARKERS = ('.lora_down', '.lora_A')
LORA_UP_MARKERS = ('.lora_up', '.lora_B')
LORA_WEIGHT_MARKERS = LORA_DOWN_MARKERS + LORA_UP_MARKERS + ('.lora_mid',)
LORA_PAIR_MARKERS = (
    ('.lora_down', '.lora_up', 'kohya'),
    ('.lora_A', '.lora_B', 'peft'),
)
DORA_MARKERS = ('.lora_magnitude_vector', '.dora_scale')


def _lora_module_from_key(key):
    """Return the module prefix for supported Kohya and PEFT LoRA keys."""
    positions = [key.find(marker) for marker in LORA_WEIGHT_MARKERS if marker in key]
    return key[:min(positions)] if positions else None


def _rank_from_weight(key, tensor):
    """Infer LoRA rank from either the down/A or up/B matrix."""
    if not isinstance(tensor, torch.Tensor) or tensor.ndim == 0:
        return None
    if any(marker in key for marker in LORA_DOWN_MARKERS):
        return tensor.size(0)
    if any(marker in key for marker in LORA_UP_MARKERS) and tensor.ndim >= 2:
        return tensor.size(1)
    return None


def _is_up_weight(key):
    return any(marker in key for marker in LORA_UP_MARKERS)


def _lora_pair_info(key):
    """Return (module, role, style) for a paired LoRA factor key."""
    for down_marker, up_marker, style in LORA_PAIR_MARKERS:
        for marker, role in ((down_marker, 'down'), (up_marker, 'up')):
            module, found, suffix = key.partition(marker)
            if found and suffix in ('', '.weight', '.default.weight'):
                return module, role, style
    return None


def _load_lora_state(path):
    """Use ComfyUI's safe loader, including its .sft and checkpoint handling."""
    from comfy.utils import load_torch_file

    return load_torch_file(path, safe_load=True)


def _resolve_output_path(filename):
    """Keep output files and symlink targets inside the configured merge folder."""
    if not isinstance(filename, str) or not filename.strip():
        raise ValueError("Choose a LoRA output filename.")
    # Check both separator conventions so workflows behave the same on Windows
    # and POSIX, and do not permit Windows drive paths or alternate data streams.
    parts = filename.replace('\\', '/').split('/')
    if (os.path.isabs(filename) or ntpath.isabs(filename)
            or ntpath.splitdrive(filename)[0] or '..' in parts
            or ':' in filename):
        raise ValueError(
            "LoRA output must be a relative filename or subfolder inside "
            "OUTPUT_DIR (krea2-merged-loras); absolute paths and '..' are not allowed."
        )
    if os.path.splitext(filename)[1].lower() not in (
        '.safetensors', '.sft', '.pt', '.pth', '.ckpt', '.bin',
    ):
        raise ValueError("Use a LoRA output extension such as .safetensors or .sft.")
    root = os.path.realpath(OUTPUT_DIR)
    output = os.path.realpath(os.path.join(root, *parts))
    try:
        inside = os.path.commonpath((root, output)) == root and output != root
    except ValueError:
        inside = False
    if not inside:
        raise ValueError("LoRA output resolves outside OUTPUT_DIR (krea2-merged-loras).")
    return output

# =============================================================================
# Krea2MergeLoadLoRA
# =============================================================================
class Krea2MergeLoadLoRA:
    """Load a single LoRA file from <ComfyUI>/models/loras.

    * `category_filter` – folder drop‑down (handled by front‑end JS)
    * `lora_name`       – file selector (full list, filtered on the client)
    """

    @classmethod
    def INPUT_TYPES(cls):
        names = folder_paths.get_filename_list("loras")
        dirs  = sorted({os.path.dirname(p) for p in names if os.path.dirname(p)})
        return {
            "required": {
                "category_filter": (["All"] + dirs,),
                "lora_name":       (names,),
            }
        }
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "Krea2 Merge/LoRA"

    def load(self, lora_name, category_filter='All'):
        # Front‑end JS handles the folder filtering; the back‑end only loads the file.
        lora_path = folder_paths.get_full_path('loras', lora_name)
        if not lora_path or not os.path.exists(lora_path):
            raise FileNotFoundError(lora_name)

        state_dict = _load_lora_state(lora_path)
        return (state_dict,)


# =============================================================================
# Krea2MergeApplyLoRA (apply to MODEL with strength; with folder filter menu)
# =============================================================================

class Krea2MergeApplyLoRA:
    """Load and apply a LoRA to a MODEL with a strength slider.
    Adds two things compared with *Krea2MergeLoadLoRA*:
      - a left-side **model** input
      - a **strength_model** slider
    The folder filtering UI is handled by the web script (same keys: `category_filter` + `lora_name`).
    """

    @classmethod
    def INPUT_TYPES(cls):
        names = folder_paths.get_filename_list("loras")
        dirs  = sorted({os.path.dirname(p) for p in names if os.path.dirname(p)})
        return {
            "required": {
                "model": ("MODEL",),
                "category_filter": (["All"] + dirs,),
                "lora_name": (names,),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"
    CATEGORY = "Krea2 Merge/LoRA"

    
    def apply(self, model, lora_name, strength_model=1.0, category_filter='All'):
        # Resolve file path
        lora_path = folder_paths.get_full_path('loras', lora_name)
        if not lora_path or not os.path.exists(lora_path):
            raise FileNotFoundError(lora_name)

        try:
            from comfy import sd as comfy_sd
        except Exception as e:
            raise ImportError("Cannot import comfy.sd helper: " + str(e))

        sd_dict = _load_lora_state(lora_path)
        new_model, _ = comfy_sd.load_lora_for_models(
            model, None, sd_dict, strength_model, 0.0,
        )
        return (new_model,)

# =============================================================================
# Krea2MergeLoRAs
# =============================================================================
class Krea2MergeLoRAs:
    """Merge Krea 2/PEFT and Kohya LoRA state dictionaries.

    * `exact_concat` is the default and composes weighted LoRAs without cross terms.
    * `legacy_linear` preserves the original approximate factor-space behavior.
    * `force_same_strength=yes` applies only to `legacy_linear`.
    * Supports both `lora_A/lora_B` (PEFT/Diffusers, including Krea 2) and
      `lora_down/lora_up` (Kohya) keys.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model1": ("MODEL",),
                "weight1": ("FLOAT", {"default": 0.50, "min": -4.0, "max": 4.0, "step": 0.01}),
                "model2": ("MODEL",),
                "weight2": ("FLOAT", {"default": 0.50, "min": -4.0, "max": 4.0, "step": 0.01}),
                "weight3": ("FLOAT", {"default": 0.00, "min": -4.0, "max": 4.0, "step": 0.01}),
                "weight4": ("FLOAT", {"default": 0.00, "min": -4.0, "max": 4.0, "step": 0.01}),
                "force_same_strength": (["no", "yes"], {
                    "default": "no",
                    "tooltip": "Legacy mode only. Ignored by exact_concat.",
                }),
                "save_dtype": (["fp16", "float", "bf16"], {"default": "fp16"}),
                "merge_mode": (["exact_concat", "legacy_linear"], {
                    "default": "exact_concat",
                    "tooltip": (
                        "exact_concat performs an exact weighted merge and supports "
                        "different ranks. legacy_linear preserves the original "
                        "approximate factor-space behavior."
                    ),
                }),
            },
            "optional": {
                "model3": ("MODEL",),
                "model4": ("MODEL",),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("merged_model",)
    FUNCTION = "merge"
    CATEGORY = "Krea2 Merge/LoRA"
    DESCRIPTION = (
        "Exactly merge two to four Krea 2/PEFT or Kohya LoRAs, including "
        "different ranks. The legacy approximate merge remains available for "
        "existing workflows."
    )

    # ---------- helpers ----------
    def _safe_scalar(self, value):
        """Return float value regardless of dtype/shape."""
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return value.float().item()
            return value.float().mean().item()
        return float(value)

    def _module_alphas(self, state_dict):
        """Collect explicit alpha values, falling back to the tensor rank."""
        alphas = {}

        for key, value in state_dict.items():
            if key.endswith('.alpha'):
                module = key[:-6]
                alpha = self._safe_scalar(value)
                if not math.isfinite(alpha) or alpha <= 0:
                    raise ValueError(f"Invalid LoRA alpha for module '{module}': {alpha}")
                alphas[module] = alpha

        # Prefer down/A because its leading dimension is always the rank.
        ordered_markers = LORA_DOWN_MARKERS + LORA_UP_MARKERS
        for marker in ordered_markers:
            for key, value in state_dict.items():
                if marker not in key:
                    continue
                module = _lora_module_from_key(key)
                rank = _rank_from_weight(key, value)
                if rank is not None and rank <= 0:
                    raise ValueError(f"Invalid LoRA rank for key '{key}': {rank}")
                if module is not None and module not in alphas and rank is not None:
                    alphas[module] = float(rank)

        return alphas

    def _paired_lora_modules(self, state_dict, input_index):
        """Collect complete A/B or down/up pairs for exact concatenation."""
        pairs = {}

        for key, tensor in state_dict.items():
            if '.lora_mid' in key:
                raise ValueError(
                    "LoCon lora_mid tensors cannot be merged correctly by either merge mode."
                )

            info = _lora_pair_info(key)
            if info is None or not isinstance(tensor, torch.Tensor):
                continue

            module, role, style = info
            pair = pairs.setdefault(module, {"style": style})
            if pair["style"] != style:
                raise ValueError(
                    f"Input model {input_index} mixes PEFT and Kohya keys for "
                    f"module '{module}', which exact_concat cannot pair safely."
                )
            if role in pair:
                raise ValueError(
                    f"Input model {input_index} has more than one {role} tensor "
                    f"for module '{module}', which exact_concat cannot pair safely."
                )
            pair[role] = tensor
            pair[f"{role}_key"] = key

        for module, pair in pairs.items():
            if "down" not in pair or "up" not in pair:
                missing = "up/B" if "up" not in pair else "down/A"
                raise ValueError(
                    f"Input model {input_index} is missing the {missing} tensor "
                    f"for module '{module}'. exact_concat needs complete LoRA pairs."
                )

            down = pair["down"]
            up = pair["up"]
            if down.ndim < 2 or up.ndim < 2:
                raise ValueError(
                    f"Invalid LoRA tensor dimensions for module '{module}': "
                    f"down/A {tuple(down.shape)}, up/B {tuple(up.shape)}."
                )
            if down.size(0) != up.size(1):
                raise ValueError(
                    f"Inconsistent LoRA rank for module '{module}': down/A has "
                    f"rank {down.size(0)}, up/B has rank {up.size(1)}."
                )
            pair["rank"] = down.size(0)

        return pairs

    def _merge_exact_concat(self, models_with_w, module_alphas_list, final_dtype,
                            force_same_strength):
        """Represent a weighted sum exactly by concatenating LoRA ranks."""
        if force_same_strength == "yes":
            print(
                "Krea2 Merge: force_same_strength is ignored by exact_concat; "
                "weights are applied directly."
            )

        paired_models = [
            self._paired_lora_modules(sd, index)
            for index, (sd, _) in enumerate(models_with_w, start=1)
        ]
        modules = sorted({module for pairs in paired_models for module in pairs})
        if not modules:
            raise ValueError(
                "exact_concat found no complete lora_A/lora_B or "
                "lora_down/lora_up pairs."
            )

        merged_sd = {}
        for module in modules:
            entries = []
            for (_, ratio), alphas, pairs in zip(
                models_with_w, module_alphas_list, paired_models
            ):
                pair = pairs.get(module)
                if pair is not None:
                    entries.append((pair, float(ratio), alphas[module]))

            styles = {pair["style"] for pair, _, _ in entries}
            if len(styles) != 1:
                raise ValueError(
                    f"Module '{module}' uses both PEFT and Kohya key styles. "
                    "exact_concat requires one consistent style per module."
                )

            first_pair = entries[0][0]
            down_shape = tuple(first_pair["down"].shape[1:])
            up_shape = (
                tuple(first_pair["up"].shape[:1])
                + tuple(first_pair["up"].shape[2:])
            )
            down_parts = []
            up_parts = []

            for pair, ratio, alpha in entries:
                down = pair["down"]
                up = pair["up"]
                current_down_shape = tuple(down.shape[1:])
                current_up_shape = tuple(up.shape[:1]) + tuple(up.shape[2:])
                if current_down_shape != down_shape or current_up_shape != up_shape:
                    raise ValueError(
                        f"Cannot exact-concat module '{module}': non-rank tensor "
                        f"dimensions differ (down/A {tuple(down.shape)}, "
                        f"up/B {tuple(up.shape)}). The LoRAs must target the "
                        "same base-model layer."
                    )

                rank = pair["rank"]
                down_parts.append(down.float())
                up_parts.append(up.float() * (ratio * alpha / rank))

            merged_down = torch.cat(down_parts, dim=0)
            merged_up = torch.cat(up_parts, dim=1)
            output_rank = merged_down.size(0)

            merged_sd[first_pair["down_key"]] = merged_down.to(dtype=final_dtype)
            merged_sd[first_pair["up_key"]] = merged_up.to(dtype=final_dtype)
            merged_sd[f"{module}.alpha"] = torch.tensor(
                float(output_rank), dtype=torch.float32
            )

        return merged_sd
    
    # ---------- main ----------
    def merge(self, model1, weight1, model2, weight2,
              weight3, weight4, force_same_strength, save_dtype,
              merge_mode="exact_concat", model3=None, model4=None):

        for slot, model, weight in ((3, model3, weight3), (4, model4, weight4)):
            if model is not None and weight == 0:
                print(
                    f"Krea2 Merge warning: model{slot} is connected but "
                    f"weight{slot} is 0, so that LoRA is ignored."
                )

        models_with_w = [(model1, weight1), (model2, weight2)]
        if model3 is not None:
            models_with_w.append((model3, weight3))
        if model4 is not None:
            models_with_w.append((model4, weight4))

        # remove zero-weight items
        models_with_w = [(sd, w) for sd, w in models_with_w if w != 0]
        if len(models_with_w) < 2:
            raise ValueError("Krea2 Merge needs at least two LoRAs with a non-zero weight.")

        # ---- gather α for each module ----
        module_alphas_list = []
        merged_base_alpha = {}

        for index, (sd, _) in enumerate(models_with_w, start=1):
            if not hasattr(sd, 'items'):
                raise TypeError(f"Input model {index} is not a LoRA state dictionary.")
            dora_key = next(
                (key for key in sd if any(marker in key for marker in DORA_MARKERS)),
                None,
            )
            if dora_key is not None:
                raise ValueError(
                    f"Input model {index} contains DoRA weights ('{dora_key}'). "
                    "DoRA merging is not supported; refusing to save a partial "
                    "or behavior-changing adapter."
                )
            if any('.lora_mid' in key for key in sd):
                raise ValueError(
                    f"Input model {index} contains LoCon lora_mid weights. "
                    "They cannot be merged correctly by either merge mode."
                )
            for key, value in sd.items():
                if key.endswith('.alpha'):
                    continue
                if _lora_pair_info(key) is None:
                    raise ValueError(
                        f"Input model {index} contains an unsupported adapter key "
                        f"('{key}'). Only lora_A/lora_B or lora_down/lora_up "
                        "factors and alpha are supported; refusing to drop adapter data."
                    )
                if not isinstance(value, torch.Tensor):
                    raise TypeError(f"LoRA factor '{key}' in input model {index} is not a tensor.")
            if not any(_lora_module_from_key(key) is not None for key in sd):
                raise ValueError(
                    f"Input model {index} has no supported LoRA weights. "
                    "Expected lora_A/lora_B or lora_down/lora_up keys."
                )
            alphas = self._module_alphas(sd)
            if not alphas:
                raise ValueError(
                    f"Input model {index} has no supported LoRA weights. "
                    "Expected lora_A/lora_B or lora_down/lora_up keys."
                )
            module_alphas_list.append(alphas)
            # accumulate
            for m, a in alphas.items():
                s, c = merged_base_alpha.get(m, (0.0, 0))
                merged_base_alpha[m] = (s + a, c + 1)

        # average to get base α
        for m, (s, c) in merged_base_alpha.items():
            merged_base_alpha[m] = s / c

        module_names = [
            {info[0] for key in sd if (info := _lora_pair_info(key)) is not None}
            for sd, _ in models_with_w
        ]
        for i, names in enumerate(module_names):
            for j in range(i + 1, len(module_names)):
                if names.isdisjoint(module_names[j]):
                    print(
                        f"Krea2 Merge warning: input models {i + 1} and {j + 1} "
                        "have no shared module names. This can mean different target "
                        "layers or incompatible key naming (for example PEFT vs Kohya). "
                        "Keys are preserved without conversion; verify that the LoRAs "
                        "use the same base model and naming convention."
                    )

        # dtype
        dtype_map = {"fp16": torch.float16, "float": torch.float32, "bf16": torch.bfloat16}
        final_dtype = dtype_map[save_dtype]

        if merge_mode == "exact_concat":
            return (
                self._merge_exact_concat(
                    models_with_w,
                    module_alphas_list,
                    final_dtype,
                    force_same_strength,
                ),
            )
        if merge_mode != "legacy_linear":
            raise ValueError(f"Unknown merge mode: {merge_mode}")

        print(
            "Krea2 Merge warning: legacy_linear is an approximate compatibility "
            "mode and introduces cross terms. Use exact_concat for an exact "
            "weighted merge."
        )

        # --- 2. Actual merging ---
        merged_sd: dict[str, torch.Tensor] = {}
        for (sd, ratio), mod_alpha in zip(models_with_w, module_alphas_list):
            if force_same_strength == "yes":
                ratio = math.copysign(math.sqrt(abs(ratio)), ratio)
            for k, tensor in sd.items():
                module = _lora_module_from_key(k)
                if k.endswith('.alpha') or module is None:
                    continue
                if not isinstance(tensor, torch.Tensor):
                    continue
                base_alpha = merged_base_alpha.get(module)
                if base_alpha is None:
                    # Auxiliary LoRA tensors such as lora_mid use the alpha of
                    # their associated down/up pair. Ignore an orphaned tensor
                    # instead of failing with a KeyError.
                    continue
                alpha_i = mod_alpha.get(module, base_alpha)
                scale = math.sqrt(alpha_i / base_alpha) * ratio
                # Keep up/B positive for a negative ratio so the composed LoRA
                # delta changes sign once, matching SuperMerger behavior.
                if _is_up_weight(k) and scale < 0:
                    scale = abs(scale)
                contrib = tensor.float() * scale
                if k in merged_sd and merged_sd[k].shape != contrib.shape:
                    raise ValueError(
                        f"Cannot merge '{k}': tensor shapes differ "
                        f"({tuple(merged_sd[k].shape)} vs {tuple(contrib.shape)}). "
                        "Use Krea 2 LoRAs with matching ranks and target modules."
                    )
                merged_sd[k] = contrib if k not in merged_sd else merged_sd[k] + contrib

        # --- 3. Write back averaged α keys ---
        for module, base_alpha in merged_base_alpha.items():
            merged_sd[f"{module}.alpha"] = torch.tensor(base_alpha, dtype=torch.float32)

        # cast
        for k in list(merged_sd.keys()):
            if not k.endswith('.alpha') and merged_sd[k].dtype != final_dtype:
                merged_sd[k] = merged_sd[k].to(dtype=final_dtype)

        return (merged_sd, )

# =============================================================================
# Krea2MergeSaveLoRA
# =============================================================================
class Krea2MergeSaveLoRA:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "merged_model": ("MODEL", ),
                "modeloutput": ("STRING", {
                    "default": "krea2_merged_lora.safetensors",
                    "tooltip": "Filename or subfolder inside krea2-merged-loras. Absolute paths and '..' are not allowed.",
                }),
                "allow_overwrite": (["no", "yes"], {"default": "no"}),
            }
        }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_path",)
    FUNCTION = "save"
    CATEGORY = "Krea2 Merge/LoRA"
    OUTPUT_NODE = True

    def save(self, merged_model, modeloutput, allow_overwrite="no"):
        modeloutput = _resolve_output_path(modeloutput)
        os.makedirs(os.path.dirname(modeloutput), exist_ok=True)

        backup_path = None
        if os.path.exists(modeloutput):
            if allow_overwrite != "yes":
                raise FileExistsError(
                    f"LoRA output already exists: {modeloutput}. "
                    "Set allow_overwrite to yes to replace it."
                )
            backup_path = f"{modeloutput}.backup-{uuid.uuid4().hex}"
            try:
                os.replace(modeloutput, backup_path)
            except OSError as error:
                raise PermissionError(
                    f"Cannot overwrite LoRA output: {modeloutput}. The file may be "
                    "open or memory-mapped by a LoRA loader. Choose another filename "
                    "or restart ComfyUI, then try again."
                ) from error

        try:
            if modeloutput.lower().endswith((".safetensors", ".sft")):
                if not safetensors_available or safe_save is None:
                    raise ImportError("pip install safetensors to save .safetensors")
                safe_save(merged_model, modeloutput)
            else:
                torch.save(merged_model, modeloutput)
        except Exception:
            if backup_path and os.path.exists(backup_path):
                try:
                    if os.path.exists(modeloutput):
                        os.remove(modeloutput)
                    os.replace(backup_path, modeloutput)
                except OSError as restore_error:
                    raise RuntimeError(
                        "Saving failed and the original file could not be restored "
                        f"automatically. Its backup is at: {backup_path}"
                    ) from restore_error
            raise
        else:
            if backup_path and os.path.exists(backup_path):
                try:
                    os.remove(backup_path)
                except OSError as cleanup_error:
                    print(f"Krea2 Merge warning: could not remove backup: {cleanup_error}")

        print(f"LoRA model saved to {modeloutput}")
        return (modeloutput,)
    
# =============================================================================
# Node registration
# =============================================================================
NODE_CLASS_MAPPINGS = {
    "Krea2Merge_LoadLoRA": Krea2MergeLoadLoRA,
    "Krea2Merge_ApplyLoRA": Krea2MergeApplyLoRA,
    "Krea2Merge_MergeLoRAs": Krea2MergeLoRAs,
    "Krea2Merge_SaveLoRA": Krea2MergeSaveLoRA,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Krea2Merge_LoadLoRA": "Krea2 Merge • Load LoRA",
    "Krea2Merge_ApplyLoRA": "Krea2 Merge • Apply LoRA",
    "Krea2Merge_MergeLoRAs": "Krea2 Merge • Merge LoRAs",
    "Krea2Merge_SaveLoRA": "Krea2 Merge • Save LoRA",
}
