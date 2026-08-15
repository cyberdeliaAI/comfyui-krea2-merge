import torch
import os
import math
import folder_paths

# -------------------------------------------------------------------
# Optional safetensors support flag (keep both names for legacy checks)
# -------------------------------------------------------------------
try:
    from safetensors.torch import load_file as safe_load, save_file as safe_save
    SAFETENSORS = True
    safetensors_available = True   # legacy alias
except ImportError:
    SAFETENSORS = False
    safetensors_available = False
    safe_load = None
    safe_save = None

# -------------------------------------------------------------------
# Build default output directory: <ComfyUI>/models/loras/krea2-merged-loras
# -------------------------------------------------------------------
lora_base_path = folder_paths.get_folder_paths("loras")[0]
OUTPUT_DIR = os.path.join(lora_base_path, "krea2-merged-loras")


LORA_DOWN_MARKERS = ('.lora_down', '.lora_A')
LORA_UP_MARKERS = ('.lora_up', '.lora_B')
LORA_WEIGHT_MARKERS = LORA_DOWN_MARKERS + LORA_UP_MARKERS + ('.lora_mid',)


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

        if lora_path.endswith('.safetensors'):
            if not SAFETENSORS or safe_load is None:
                raise ImportError('pip install safetensors to load .safetensors')
            state_dict = safe_load(lora_path, device='cpu')
        else:
            state_dict = torch.load(lora_path, map_location='cpu')
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

        # Helper: load to state-dict (for Comfy versions that require dict)
        def _load_state(p):
            if p.endswith('.safetensors'):
                if not SAFETENSORS or safe_load is None:
                    raise ImportError('pip install safetensors to load .safetensors')
                return safe_load(p, device='cpu')
            else:
                import torch
                return torch.load(p, map_location='cpu')

        # Try both calling conventions for better compatibility.
        # 1) Prefer passing dict (newer implementations often expect a dict)
        last_err = None
        try:
            sd_dict = _load_state(lora_path)
            new_model, _ = comfy_sd.load_lora_for_models(model, None, sd_dict, strength_model, 0.0)
            return (new_model,)
        except Exception as e:
            last_err = e
            # Fallback 2) pass file path string
            try:
                new_model, _ = comfy_sd.load_lora_for_models(model, None, lora_path, strength_model, 0.0)
                return (new_model,)
            except Exception as e2:
                raise RuntimeError(f"load_lora_for_models failed (dict -> {last_err}); (path -> {e2})")

# =============================================================================
# Krea2MergeLoRAs
# =============================================================================
class Krea2MergeLoRAs:
    """Merge Krea 2/PEFT and Kohya LoRA state dictionaries.

    * **Order‑independent** – A+B == B+A when ratios are the same.
    * `force_same_strength=yes`  ⇒   ratio → √ratio (matches Web‑UI "Strength").
    * Per‑module scaling by √(αᵢ / avgα) for more balanced feature retention.
    * Supports both `lora_A/lora_B` (PEFT/Diffusers, including Krea 2) and
      `lora_down/lora_up` (Kohya) keys.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model1": ("MODEL",),
                "weight1": ("FLOAT", {"default": 1.00,"step": 0.01}),
                "model2": ("MODEL",),
                "weight2": ("FLOAT", {"default": 1.00,"step": 0.01}),
                "weight3": ("FLOAT", {"default": 0.00,"step": 0.01}),
                "weight4": ("FLOAT", {"default": 0.00,"step": 0.01}),
                "force_same_strength": (["no", "yes"], {"default": "no"}),
                "save_dtype": (["fp16", "float", "bf16"], {"default": "fp16"}),
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
    
    # ---------- main ----------
    def merge(self, model1, weight1, model2, weight2,
              weight3, weight4, force_same_strength, save_dtype,
              model3=None, model4=None):

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

        # dtype
        dtype_map = {"fp16": torch.float16, "float": torch.float32, "bf16": torch.bfloat16}
        final_dtype = dtype_map[save_dtype]

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
            merged_sd[f"{module}.alpha"] = torch.tensor(base_alpha, dtype=final_dtype)

        # cast
        for k in list(merged_sd.keys()):
            if merged_sd[k].dtype != final_dtype:
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
                "modeloutput": ("STRING", {"default": "krea2_merged_lora.safetensors"}),
            }
        }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_path",)
    FUNCTION = "save"
    CATEGORY = "Krea2 Merge/LoRA"
    OUTPUT_NODE = True

    def save(self, merged_model, modeloutput):
        if not os.path.isabs(modeloutput):
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            modeloutput = os.path.join(OUTPUT_DIR, modeloutput)
        if modeloutput.endswith('.safetensors'):
            if not safetensors_available or safe_save is None:
                raise ImportError('pip install safetensors to save .safetensors')
            safe_save(merged_model, modeloutput)
        else:
            torch.save(merged_model, modeloutput)
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
