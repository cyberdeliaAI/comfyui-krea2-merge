import os
import sys
import tempfile
import types
import unittest

import torch


folder_paths = types.ModuleType("folder_paths")
folder_paths.get_folder_paths = lambda _kind: [tempfile.gettempdir()]
folder_paths.get_filename_list = lambda _kind: []
folder_paths.get_full_path = lambda _kind, name: os.path.join(tempfile.gettempdir(), name)
sys.modules.setdefault("folder_paths", folder_paths)

from mergetools.merge_lora_tools import Krea2MergeLoRAs


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

    def test_rejects_unsupported_adapter_state_dict(self):
        key = f"{self.module}.lora_magnitude_vector.weight"
        unsupported = {key: torch.ones(4)}

        with self.assertRaisesRegex(ValueError, "no supported LoRA weights"):
            self.merge(unsupported, unsupported)


if __name__ == "__main__":
    unittest.main()
