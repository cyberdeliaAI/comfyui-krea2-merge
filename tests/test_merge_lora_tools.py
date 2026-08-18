import os
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import torch


folder_paths = types.ModuleType("folder_paths")
folder_paths.get_folder_paths = lambda _kind: [tempfile.gettempdir()]
folder_paths.get_filename_list = lambda _kind: []
folder_paths.get_full_path = lambda _kind, name: os.path.join(tempfile.gettempdir(), name)
sys.modules.setdefault("folder_paths", folder_paths)

from mergetools.merge_lora_tools import Krea2MergeLoRAs, Krea2MergeSaveLoRA


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
        )[0]

    def exact_merge(self, model1, model2, weight1=1.0, weight2=1.0,
                    force_same_strength="no"):
        return self.merger.merge(
            model1=model1,
            weight1=weight1,
            model2=model2,
            weight2=weight2,
            weight3=0.0,
            weight4=0.0,
            force_same_strength=force_same_strength,
            save_dtype="float",
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

    def test_rejects_unsupported_adapter_state_dict(self):
        key = f"{self.module}.lora_magnitude_vector.weight"
        unsupported = {key: torch.ones(4)}

        with self.assertRaisesRegex(ValueError, "no supported LoRA weights"):
            self.merge(unsupported, unsupported)

    def test_save_refuses_to_overwrite_existing_file_by_default(self):
        saver = Krea2MergeSaveLoRA()
        with tempfile.TemporaryDirectory() as directory:
            output = os.path.join(directory, "existing.pt")
            with open(output, "wb") as existing_file:
                existing_file.write(b"keep")
            with self.assertRaisesRegex(FileExistsError, "allow_overwrite"):
                saver.save({}, output, "no")

    def test_save_overwrites_when_enabled(self):
        saver = Krea2MergeSaveLoRA()
        with tempfile.TemporaryDirectory() as directory:
            output = os.path.join(directory, "existing.pt")
            with open(output, "wb") as existing_file:
                existing_file.write(b"replace")

            def write_replacement(_model, path):
                with open(path, "wb") as replacement_file:
                    replacement_file.write(b"replacement")

            with patch(
                "mergetools.merge_lora_tools.torch.save",
                side_effect=write_replacement,
            ) as save_mock:
                result = saver.save({}, output, "yes")

            save_mock.assert_called_once_with({}, output)
            self.assertEqual(result, (output,))
            with open(output, "rb") as saved_file:
                self.assertEqual(saved_file.read(), b"replacement")
            self.assertFalse(any(".backup-" in name for name in os.listdir(directory)))

    def test_save_restores_original_when_overwrite_fails(self):
        saver = Krea2MergeSaveLoRA()
        with tempfile.TemporaryDirectory() as directory:
            output = os.path.join(directory, "existing.pt")
            with open(output, "wb") as existing_file:
                existing_file.write(b"original")

            with patch(
                "mergetools.merge_lora_tools.torch.save",
                side_effect=RuntimeError("simulated save failure"),
            ):
                with self.assertRaisesRegex(RuntimeError, "simulated save failure"):
                    saver.save({}, output, "yes")

            with open(output, "rb") as restored_file:
                self.assertEqual(restored_file.read(), b"original")
            self.assertFalse(any(".backup-" in name for name in os.listdir(directory)))

    def test_save_node_is_registered_as_output(self):
        self.assertTrue(Krea2MergeSaveLoRA.OUTPUT_NODE)


if __name__ == "__main__":
    unittest.main()
