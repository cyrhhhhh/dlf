import unittest
from types import SimpleNamespace

import torch

from trains.singleTask.DLF import (
    _accumulation_group_size,
    _should_optimizer_step,
)
from trains.singleTask.model.SerialDLF import DLF


def make_args():
    return SimpleNamespace(
        use_ccdr=True,
        use_bert=False,
        seq_lens=[8, 8, 8],
        feature_dims=[6, 4, 5],
        dst_feature_dim_nheads=[4, 2],
        nlevels=1,
        attn_dropout=0.0,
        attn_dropout_a=0.0,
        attn_dropout_v=0.0,
        relu_dropout=0.0,
        embed_dropout=0.0,
        res_dropout=0.0,
        output_dropout=0.0,
        text_dropout=0.0,
        attn_mask=False,
        conv1d_kernel_size_l=3,
        conv1d_kernel_size_a=3,
        conv1d_kernel_size_v=3,
        ccdr_dropout=0.0,
        cred_route_temperature=1.0,
    )


class SerialDLFTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = DLF(make_args()).eval()
        self.padding_mask = torch.tensor([
            [False, False, False, False, False, True, True, True],
            [False, False, False, True, True, True, True, True],
        ])
        self.text = torch.randn(2, 8, 6)
        self.audio = torch.randn(2, 8, 4)
        self.vision = torch.randn(2, 8, 5)

    def test_serial_output_and_padding_invariance(self):
        with torch.no_grad():
            output = self.model(
                self.text, self.audio, self.vision, self.padding_mask
            )
            perturbed_text = self.text.masked_fill(
                self.padding_mask.unsqueeze(-1), 1000.0
            )
            perturbed_audio = self.audio.masked_fill(
                self.padding_mask.unsqueeze(-1), -1000.0
            )
            perturbed_vision = self.vision.masked_fill(
                self.padding_mask.unsqueeze(-1), 500.0
            )
            perturbed_output = self.model(
                perturbed_text,
                perturbed_audio,
                perturbed_vision,
                self.padding_mask,
            )

        expected = (
            output['consensus_logit']
            + output['support_delta']
            + output['conflict_delta']
        )
        self.assertTrue(torch.allclose(output['output_logit'], expected))
        self.assertTrue(torch.allclose(
            output['output_logit'], perturbed_output['output_logit'], atol=1e-6
        ))
        self.assertFalse(hasattr(self.model, 'trans_l_with_a'))
        for mask in output['projected_padding_masks']:
            self.assertEqual(mask.shape, (2, 6))
            self.assertTrue((~mask).any(dim=1).all())

    def test_serial_modules_receive_gradients(self):
        self.model.train()
        output = self.model(
            self.text, self.audio, self.vision, self.padding_mask
        )
        loss = (
            output['output_logit'].sum()
            + output['logits_l_shared'].sum()
            + output['logits_a_shared'].sum()
            + output['logits_v_shared'].sum()
        )
        loss.backward()

        self.assertTrue(any(
            parameter.grad is not None
            for parameter in self.model.encoder_c.parameters()
        ))
        self.assertTrue(any(
            parameter.grad is not None
            for parameter in self.model.encoder_s_l.parameters()
        ))
        self.assertTrue(any(
            parameter.grad is not None
            for parameter in self.model.ccdr.parameters()
        ))


class AccumulationBoundaryTest(unittest.TestCase):
    def test_partial_tail_group_is_updated(self):
        total_batches = 23
        accumulation_steps = 10
        step_batches = [
            index + 1
            for index in range(total_batches)
            if _should_optimizer_step(
                index, total_batches, accumulation_steps
            )
        ]
        group_sizes = [
            _accumulation_group_size(
                index, total_batches, accumulation_steps
            )
            for index in range(total_batches)
        ]

        self.assertEqual(step_batches, [10, 20, 23])
        self.assertEqual(group_sizes, [10] * 20 + [3] * 3)


if __name__ == '__main__':
    unittest.main()