import os
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import torch


folder_paths = types.ModuleType("folder_paths")
folder_paths.get_folder_paths = lambda _kind: [tempfile.gettempdir()]
folder_paths.get_filename_list = lambda _kind: []
folder_paths.get_full_path = lambda _kind, name: os.path.join(tempfile.gettempdir(), name)
sys.modules.setdefault("folder_paths", folder_paths)

from mergetools.merge_lora_tools import (
    Krea2MergeApplyLoRA, Krea2MergeLoadLoRA, Krea2MergeLoRAs, Krea2MergeSaveLoRA,
)


class Krea2MergeLoRAsTests(unittest.TestCase):
    def setUp(self):
        self.merger = Krea2MergeLoRAs()
        self.module = "diffusion_model.blocks.0.attn.gate"

    def merge(self, model1, model2, weight1=1.0, weight2=1.0):
        return self.merger.merge(
            model1=model1,
            weight1=weight1,
            model2=model2,
            weight2=weight2,
            weight3=0.0,
            weight4=0.0,
            force_same_strength="no",
            save_dtype="float",
            merge_mode="legacy_linear",
        )[0]

    def exact_merge(self, model1, model2, weight1=1.0, weight2=1.0,
                    force_same_strength="no", save_dtype="float"):
        return self.merger.merge(
            model1=model1,
            weight1=weight1,
            model2=model2,
            weight2=weight2,
            weight3=0.0,
            weight4=0.0,
            force_same_strength=force_same_strength,
            save_dtype=save_dtype,
            merge_mode="exact_concat",
        )[0]

    def peft_model(self, a_value, b_value, rank=2):
        return {
            f"{self.module}.lora_A.weight": torch.full((rank, 3), a_value),
            f"{self.module}.lora_B.weight": torch.full((4, rank), b_value),
        }

    def test_merges_krea2_peft_keys_and_infers_alpha_from_rank(self):
        merged = self.merge(self.peft_model(1.0, 1.0), self.peft_model(2.0, 3.0))

        torch.testing.assert_close(
            merged[f"{self.module}.lora_A.weight"], torch.full((2, 3), 3.0)
        )
        torch.testing.assert_close(
            merged[f"{self.module}.lora_B.weight"], torch.full((4, 2), 4.0)
        )
        self.assertEqual(merged[f"{self.module}.alpha"].item(), 2.0)

    def test_merge_is_order_independent_for_equal_weights(self):
        first = self.peft_model(1.0, 2.0)
        second = self.peft_model(3.0, 4.0)

        forward = self.merge(first, second)
        reverse = self.merge(second, first)

        self.assertEqual(forward.keys(), reverse.keys())
        for key in forward:
            torch.testing.assert_close(forward[key], reverse[key])

    def test_negative_weight_changes_sign_on_a_but_not_b(self):
        merged = self.merge(
            self.peft_model(1.0, 1.0),
            self.peft_model(2.0, 3.0),
            weight2=-1.0,
        )

        torch.testing.assert_close(
            merged[f"{self.module}.lora_A.weight"], torch.full((2, 3), -1.0)
        )
        torch.testing.assert_close(
            merged[f"{self.module}.lora_B.weight"], torch.full((4, 2), 4.0)
        )

    def test_retains_kohya_down_up_support(self):
        down = f"{self.module}.lora_down.weight"
        up = f"{self.module}.lora_up.weight"
        alpha = f"{self.module}.alpha"
        first = {
            down: torch.ones((2, 3)),
            up: torch.ones((4, 2)),
            alpha: torch.tensor(2.0),
        }
        second = {
            down: torch.full((2, 3), 2.0),
            up: torch.full((4, 2), 2.0),
            alpha: torch.tensor(2.0),
        }

        merged = self.merge(first, second)

        torch.testing.assert_close(merged[down], torch.full((2, 3), 3.0))
        torch.testing.assert_close(merged[up], torch.full((4, 2), 3.0))
        self.assertEqual(merged[alpha].item(), 2.0)

    def test_falls_back_to_b_matrix_when_a_is_missing(self):
        key = f"{self.module}.lora_B.weight"
        first = {key: torch.ones((4, 2))}
        second = {key: torch.full((4, 2), 2.0)}

        merged = self.merge(first, second)

        torch.testing.assert_close(merged[key], torch.full((4, 2), 3.0))
        self.assertEqual(merged[f"{self.module}.alpha"].item(), 2.0)

    def test_reports_mismatched_krea2_ranks_clearly(self):
        with self.assertRaisesRegex(ValueError, "tensor shapes differ"):
            self.merge(self.peft_model(1.0, 1.0, rank=2), self.peft_model(1.0, 1.0, rank=4))

    def test_exact_concat_supports_different_ranks(self):
        first = self.peft_model(1.0, 2.0, rank=2)
        second = self.peft_model(3.0, 4.0, rank=4)

        merged = self.exact_merge(first, second, weight1=0.5, weight2=0.25)

        down_key = f"{self.module}.lora_A.weight"
        up_key = f"{self.module}.lora_B.weight"
        self.assertEqual(tuple(merged[down_key].shape), (6, 3))
        self.assertEqual(tuple(merged[up_key].shape), (4, 6))
        self.assertEqual(merged[f"{self.module}.alpha"].item(), 6.0)

        expected_delta = (
            0.5 * (first[up_key] @ first[down_key])
            + 0.25 * (second[up_key] @ second[down_key])
        )
        output_scale = merged[f"{self.module}.alpha"] / merged[down_key].size(0)
        actual_delta = output_scale * (merged[up_key] @ merged[down_key])
        torch.testing.assert_close(actual_delta, expected_delta)

    def test_exact_concat_is_the_default_merge_mode(self):
        first = self.peft_model(1.0, 2.0, rank=2)
        second = self.peft_model(3.0, 4.0, rank=4)

        merged = self.merger.merge(
            model1=first,
            weight1=0.5,
            model2=second,
            weight2=0.5,
            weight3=0.0,
            weight4=0.0,
            force_same_strength="no",
            save_dtype="float",
        )[0]

        self.assertEqual(
            tuple(merged[f"{self.module}.lora_A.weight"].shape),
            (6, 3),
        )
        self.assertEqual(merged[f"{self.module}.alpha"].item(), 6.0)

    def test_ui_allows_negative_weights_and_defaults_to_exact_concat(self):
        required = self.merger.INPUT_TYPES()["required"]

        for name in ("weight1", "weight2", "weight3", "weight4"):
            options = required[name][1]
            self.assertEqual(options["min"], -4.0)
            self.assertEqual(options["max"], 4.0)
        self.assertEqual(required["weight1"][1]["default"], 0.5)
        self.assertEqual(required["weight2"][1]["default"], 0.5)
        self.assertEqual(required["merge_mode"][1]["default"], "exact_concat")

    def test_warns_when_connected_optional_lora_has_zero_weight(self):
        with patch("builtins.print") as print_mock:
            self.merger.merge(
                model1=self.peft_model(1.0, 1.0),
                weight1=0.5,
                model2=self.peft_model(2.0, 2.0),
                weight2=0.5,
                model3=self.peft_model(3.0, 3.0),
                weight3=0.0,
                model4=None,
                weight4=0.0,
                force_same_strength="no",
                save_dtype="float",
            )

        self.assertTrue(
            any("model3 is connected" in call.args[0] for call in print_mock.call_args_list)
        )

    def test_exact_concat_uses_explicit_alpha_and_negative_weights(self):
        first = self.peft_model(1.0, 2.0, rank=2)
        second = self.peft_model(3.0, 4.0, rank=4)
        first[f"{self.module}.alpha"] = torch.tensor(1.0)
        second[f"{self.module}.alpha"] = torch.tensor(2.0)

        merged = self.exact_merge(first, second, weight1=0.5, weight2=-0.25)

        down_key = f"{self.module}.lora_A.weight"
        up_key = f"{self.module}.lora_B.weight"
        expected_delta = (
            0.5 * (1.0 / 2.0) * (first[up_key] @ first[down_key])
            - 0.25 * (2.0 / 4.0) * (second[up_key] @ second[down_key])
        )
        actual_delta = merged[up_key] @ merged[down_key]
        torch.testing.assert_close(actual_delta, expected_delta)

    def test_exact_concat_result_is_equivalent_when_inputs_are_reversed(self):
        first = self.peft_model(1.0, 2.0, rank=2)
        second = self.peft_model(3.0, 4.0, rank=4)
        forward = self.exact_merge(first, second, weight1=0.5, weight2=0.25)
        reverse = self.exact_merge(second, first, weight1=0.25, weight2=0.5)

        down_key = f"{self.module}.lora_A.weight"
        up_key = f"{self.module}.lora_B.weight"
        torch.testing.assert_close(
            forward[up_key] @ forward[down_key],
            reverse[up_key] @ reverse[down_key],
        )

    def test_exact_concat_rejects_incomplete_pairs(self):
        incomplete = {f"{self.module}.lora_A.weight": torch.ones((2, 3))}
        with self.assertRaisesRegex(ValueError, "complete LoRA pairs"):
            self.exact_merge(incomplete, self.peft_model(1.0, 1.0))

    def test_random_matrix_deltas_match_with_mixed_ranks_alpha_and_negative_weights(self):
        for seed in (5, 21, 109):
            for down_name, up_name in (('lora_A', 'lora_B'), ('lora_down', 'lora_up')):
                for convolution in (False, True):
                    with self.subTest(seed=seed, style=down_name, convolution=convolution):
                        generator = torch.Generator().manual_seed(seed)
                        down_key = f"{self.module}.{down_name}.weight"
                        up_key = f"{self.module}.{up_name}.weight"
                        alpha_key = f"{self.module}.alpha"
                        models = []
                        for rank, alpha in ((4, 8.0), (16, 3.0)):
                            down_shape = (rank, 3, 3, 3) if convolution else (rank, 9)
                            up_shape = (7, rank, 1, 1) if convolution else (7, rank)
                            models.append({
                                down_key: torch.randn(down_shape, generator=generator),
                                up_key: torch.randn(up_shape, generator=generator),
                                alpha_key: torch.tensor(alpha),
                            })
                        merged = self.exact_merge(*models, weight1=0.7, weight2=-0.3)
                        expected = sum(
                            weight * model[alpha_key].item() / model[down_key].shape[0]
                            * (model[up_key].double().flatten(1) @ model[down_key].double().flatten(1))
                            for model, weight in zip(models, (0.7, -0.3))
                        )
                        actual = (merged[alpha_key].item() / merged[down_key].shape[0]
                                  * (merged[up_key].double().flatten(1)
                                     @ merged[down_key].double().flatten(1)))
                        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
                        self.assertEqual(merged[down_key].shape[0], 20)

    def test_bf16_output_keeps_exact_rank_259_alpha_after_save(self):
        from safetensors.torch import load_file

        first = self.peft_model(1.0, 1.0, rank=128)
        second = self.peft_model(1.0, 1.0, rank=131)
        merged = self.exact_merge(first, second, save_dtype="bf16")
        with tempfile.TemporaryDirectory() as directory, patch(
            "mergetools.merge_lora_tools.OUTPUT_DIR", directory,
        ):
            path = Krea2MergeSaveLoRA().save(merged, "rank259.safetensors")[0]
            saved = load_file(path)
        self.assertEqual(saved[f"{self.module}.alpha"].dtype, torch.float32)
        self.assertEqual(saved[f"{self.module}.alpha"].item(), 259.0)
        self.assertEqual(saved[f"{self.module}.lora_A.weight"].dtype, torch.bfloat16)
        actual = (saved[f"{self.module}.alpha"] / 259
                  * (saved[f"{self.module}.lora_B.weight"].float()
                     @ saved[f"{self.module}.lora_A.weight"].float()))
        torch.testing.assert_close(actual, torch.full((4, 3), 259.0))

    def test_legacy_bf16_keeps_alpha_in_float32(self):
        merged = self.merger.merge(
            self.peft_model(1.0, 1.0, rank=259), 0.5,
            self.peft_model(1.0, 1.0, rank=259), 0.5,
            0.0, 0.0, "no", "bf16", merge_mode="legacy_linear",
        )[0]
        self.assertEqual(merged[f"{self.module}.alpha"].dtype, torch.float32)
        self.assertEqual(merged[f"{self.module}.alpha"].item(), 259.0)
        self.assertEqual(merged[f"{self.module}.lora_B.weight"].dtype, torch.bfloat16)

    def test_rejects_extra_adapter_data_in_both_modes(self):
        for suffix in ("diff", "diff_b", "hada_w1_a", "lokr_w1", "w_norm",
                       "b_norm", "set_weight", "lora_A.bias", "unknown_tensor"):
            for merge in (self.merge, self.exact_merge):
                with self.subTest(suffix=suffix, mode=merge.__name__):
                    first = self.peft_model(1.0, 1.0)
                    first[f"{self.module}.{suffix}"] = torch.ones(4)
                    with self.assertRaisesRegex(ValueError, "unsupported adapter key"):
                        merge(first, self.peft_model(1.0, 1.0))

    def test_default_named_peft_factors_remain_supported(self):
        first = {key.replace('.weight', '.default.weight'): value
                 for key, value in self.peft_model(1.0, 1.0).items()}
        merged = self.exact_merge(first, first)
        self.assertIn(f"{self.module}.lora_A.default.weight", merged)

    def test_disjoint_modules_warn_but_preserve_both_weighted_deltas(self):
        first = self.peft_model(1.0, 2.0)
        second = {key.replace('blocks.0.', 'blocks.1.'): value
                  for key, value in self.peft_model(3.0, 4.0).items()}
        with patch("builtins.print") as print_mock:
            merged = self.exact_merge(first, second, weight1=0.7, weight2=-0.3)
        self.assertTrue(any("no shared module names" in str(call)
                            for call in print_mock.call_args_list))
        for model, weight in ((first, 0.7), (second, -0.3)):
            down, up = model
            torch.testing.assert_close(merged[up] @ merged[down], weight * (model[up] @ model[down]))

    def test_overlapping_modules_do_not_trigger_disjoint_warning(self):
        with patch("builtins.print") as print_mock:
            self.exact_merge(self.peft_model(1.0, 2.0), self.peft_model(3.0, 4.0))
        self.assertFalse(any("no shared module names" in str(call)
                             for call in print_mock.call_args_list))

    def test_rejects_unsupported_adapter_state_dict(self):
        key = f"{self.module}.lora_magnitude_vector.weight"
        unsupported = {key: torch.ones(4)}

        with self.assertRaisesRegex(ValueError, "DoRA merging is not supported"):
            self.merge(unsupported, unsupported)

    def test_rejects_dora_instead_of_silently_dropping_it(self):
        dora = self.peft_model(1.0, 1.0)
        dora[f"{self.module}.lora_magnitude_vector"] = torch.ones(4)

        with self.assertRaisesRegex(ValueError, "DoRA merging is not supported"):
            self.exact_merge(dora, self.peft_model(1.0, 1.0))

    def test_rejects_locon_mid_instead_of_scaling_it_incorrectly(self):
        locon = self.peft_model(1.0, 1.0)
        locon[f"{self.module}.lora_mid.weight"] = torch.ones((2, 2, 1, 1))

        with self.assertRaisesRegex(ValueError, "cannot be merged correctly"):
            self.merge(locon, self.peft_model(1.0, 1.0))

    def test_save_refuses_to_overwrite_existing_file_by_default(self):
        saver = Krea2MergeSaveLoRA()
        with tempfile.TemporaryDirectory() as directory, patch(
            "mergetools.merge_lora_tools.OUTPUT_DIR", directory,
        ):
            output = os.path.realpath(os.path.join(directory, "existing.pt"))
            with open(output, "wb") as existing_file:
                existing_file.write(b"keep")
            with self.assertRaisesRegex(FileExistsError, "allow_overwrite"):
                saver.save({}, "existing.pt", "no")

    def test_save_overwrites_when_enabled(self):
        saver = Krea2MergeSaveLoRA()
        with tempfile.TemporaryDirectory() as directory, patch(
            "mergetools.merge_lora_tools.OUTPUT_DIR", directory,
        ):
            output = os.path.realpath(os.path.join(directory, "existing.pt"))
            with open(output, "wb") as existing_file:
                existing_file.write(b"replace")

            def write_replacement(_model, path):
                with open(path, "wb") as replacement_file:
                    replacement_file.write(b"replacement")

            with patch(
                "mergetools.merge_lora_tools.torch.save",
                side_effect=write_replacement,
            ) as save_mock:
                result = saver.save({}, "existing.pt", "yes")

            save_mock.assert_called_once_with({}, output)
            self.assertEqual(result, (output,))
            with open(output, "rb") as saved_file:
                self.assertEqual(saved_file.read(), b"replacement")
            self.assertFalse(any(".backup-" in name for name in os.listdir(directory)))

    def test_save_restores_original_when_overwrite_fails(self):
        saver = Krea2MergeSaveLoRA()
        with tempfile.TemporaryDirectory() as directory, patch(
            "mergetools.merge_lora_tools.OUTPUT_DIR", directory,
        ):
            output = os.path.realpath(os.path.join(directory, "existing.pt"))
            with open(output, "wb") as existing_file:
                existing_file.write(b"original")

            with patch(
                "mergetools.merge_lora_tools.torch.save",
                side_effect=RuntimeError("simulated save failure"),
            ):
                with self.assertRaisesRegex(RuntimeError, "simulated save failure"):
                    saver.save({}, "existing.pt", "yes")

            with open(output, "rb") as restored_file:
                self.assertEqual(restored_file.read(), b"original")
            self.assertFalse(any(".backup-" in name for name in os.listdir(directory)))

    def test_save_node_is_registered_as_output(self):
        self.assertTrue(Krea2MergeSaveLoRA.OUTPUT_NODE)


class Krea2MergeFileTests(unittest.TestCase):
    def setUp(self):
        from safetensors.torch import load_file

        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.root = os.path.realpath(self.tempdir.name)
        self.output_dir = os.path.join(self.root, "merged")
        patcher = patch("mergetools.merge_lora_tools.OUTPUT_DIR", self.output_dir)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.state = {"test.lora_A.weight": torch.arange(6.0).reshape(2, 3)}
        self.saver = Krea2MergeSaveLoRA()

        # Exercise real safetensors/checkpoint files while keeping ComfyUI's
        # model dependencies out of these unit tests. Verify its safe-load API.
        def load_file_safely(path, *, safe_load):
            self.assertTrue(safe_load)
            if path.lower().endswith((".safetensors", ".sft")):
                return load_file(path, device="cpu")
            return torch.load(path, map_location="cpu", weights_only=True)

        utils = types.ModuleType("comfy.utils")
        utils.load_torch_file = Mock(side_effect=load_file_safely)
        sd = types.ModuleType("comfy.sd")
        sd.load_lora_for_models = Mock(return_value=("patched-model", None))
        comfy = types.ModuleType("comfy")
        comfy.utils, comfy.sd = utils, sd
        self.utils, self.sd = utils, sd
        # Restore only our three mock modules. Restoring all of sys.modules
        # would unload lazy PyTorch modules and re-register their operators.
        for name, module in {"comfy": comfy, "comfy.utils": utils, "comfy.sd": sd}.items():
            if name in sys.modules:
                self.addCleanup(sys.modules.__setitem__, name, sys.modules[name])
            else:
                self.addCleanup(sys.modules.pop, name, None)
            sys.modules[name] = module

    def test_load_and_apply_use_safe_loader_for_supported_files(self):
        for index, extension in enumerate(("safetensors", "sft", "SFT", "pt")):
            with self.subTest(extension=extension):
                path = self.saver.save(self.state, f"subfolder/merged_{index}.{extension}")[0]
                with patch("mergetools.merge_lora_tools.folder_paths.get_full_path", return_value=path):
                    loaded = Krea2MergeLoadLoRA().load("selected-lora")[0]
                    result = Krea2MergeApplyLoRA().apply("model", "selected-lora", 0.7)
                torch.testing.assert_close(loaded["test.lora_A.weight"], self.state["test.lora_A.weight"])
                self.utils.load_torch_file.assert_called_with(path, safe_load=True)
                args = self.sd.load_lora_for_models.call_args.args
                self.assertEqual(args[:2], ("model", None))
                torch.testing.assert_close(args[2]["test.lora_A.weight"], self.state["test.lora_A.weight"])
                self.assertEqual(args[3:], (0.7, 0.0))
                self.assertEqual(result, ("patched-model",))

    def test_save_rejects_escape_paths_before_overwriting_anything(self):
        outside = os.path.join(self.root, "outside.pt")
        with open(outside, "wb") as handle:
            handle.write(b"original")
        filenames = (outside, "../outside.pt", "..\\outside.pt", "sub/../../outside.pt",
                     "C:\\outside.pt", "C:outside.pt", "\\\\server\\share\\outside.pt",
                     "inside.pt:stream.pt", "note.txt", "", " ")
        for name in filenames:
            with self.subTest(filename=name), self.assertRaises(ValueError):
                self.saver.save(self.state, name, "yes")
        with open(outside, "rb") as handle:
            self.assertEqual(handle.read(), b"original")
        self.assertFalse(os.path.exists(self.output_dir))

    def test_save_rejects_symlink_escapes(self):
        os.makedirs(self.output_dir)
        outside = os.path.join(self.root, "outside.pt")
        with open(outside, "wb") as handle:
            handle.write(b"original")
        try:
            os.symlink(self.root, os.path.join(self.output_dir, "escape"), target_is_directory=True)
            os.symlink(outside, os.path.join(self.output_dir, "linked.pt"))
        except OSError as error:
            self.skipTest(f"Symlinks unavailable: {error}")
        for name in ("escape/outside.pt", "linked.pt"):
            with self.subTest(filename=name), self.assertRaisesRegex(ValueError, "outside OUTPUT_DIR"):
                self.saver.save(self.state, name, "yes")
        with open(outside, "rb") as handle:
            self.assertEqual(handle.read(), b"original")

    def test_save_accepts_windows_style_relative_subfolders(self):
        path = self.saver.save(self.state, "styles\\merged.safetensors")[0]
        self.assertEqual(path, os.path.join(self.output_dir, "styles", "merged.safetensors"))
        self.assertTrue(os.path.isfile(path))

    def test_safetensors_overwrite_preserves_backup_behavior(self):
        from safetensors.torch import load_file

        path = self.saver.save(self.state, "merged.safetensors")[0]
        replacement = {key: value * 2 for key, value in self.state.items()}
        self.saver.save(replacement, "merged.safetensors", "yes")
        torch.testing.assert_close(load_file(path)["test.lora_A.weight"], replacement["test.lora_A.weight"])
        self.assertEqual(os.listdir(self.output_dir), ["merged.safetensors"])

    def test_failed_safe_load_is_not_retried_through_another_api(self):
        path = self.saver.save(self.state, "merged.sft")[0]
        self.utils.load_torch_file.side_effect = ValueError("invalid checkpoint")
        with patch("mergetools.merge_lora_tools.folder_paths.get_full_path", return_value=path):
            with self.assertRaisesRegex(ValueError, "invalid checkpoint"):
                Krea2MergeApplyLoRA().apply("model", "merged.sft")
        self.sd.load_lora_for_models.assert_not_called()


if __name__ == "__main__":
    unittest.main()
