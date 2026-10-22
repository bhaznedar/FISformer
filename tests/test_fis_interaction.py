"""Unit tests for the FIS Interaction layer.  Run from the repository root:  python -m pytest tests"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from layers.FIS_Interaction import FISInteraction, FisAttention  # noqa: E402
from layers.SelfAttention_Family import AttentionLayer  # noqa: E402


def _qkv(B=2, N=7, H=8, E=16):
    torch.manual_seed(0)
    return [torch.randn(B, N, H, E) for _ in range(3)]


def test_output_shape_and_cpu():
    q, k, v = _qkv()
    out, attn = FISInteraction(input_dim=16, num_head=8)(q, k, v, None)
    assert out.shape == v.shape and attn is None


def test_weights_sum_to_one_over_tokens():
    q, k, v = _qkv()
    layer = FISInteraction(input_dim=16, num_head=8, output_attention=True)
    _, A = layer(q, k, v, None)
    assert torch.allclose(A.sum(dim=1), torch.ones_like(A.sum(dim=1)), atol=1e-5)


def test_gradients_reach_all_fis_parameters():
    q, k, v = _qkv()
    layer = FISInteraction(input_dim=16, num_head=8)
    layer(q, k, v, None)[0].pow(2).mean().backward()
    for name in FISInteraction.PARAM_NAMES:
        g = getattr(layer, name).grad
        assert g is not None and torch.isfinite(g).all(), name


def test_inside_attention_layer_with_other_head_counts():
    for n_heads in (1, 4, 8):
        d_model = 64
        layer = AttentionLayer(FISInteraction(input_dim=d_model // n_heads, num_head=n_heads), d_model, n_heads)
        x = torch.randn(3, 21, d_model)
        out, _ = layer(x, x, x, attn_mask=None)
        assert out.shape == x.shape


def test_record_and_save_maps(tmp_path):
    q, k, v = _qkv()
    layer = FISInteraction(input_dim=16, num_head=8, record_maps=True).eval()
    layer(q, k, v, None)
    layer.save_maps(str(tmp_path))
    names = set(os.listdir(tmp_path))
    assert 'attention_scores_Local.npy' in names and 'rule_weights.npy' in names


def test_backward_compatible_alias():
    assert FisAttention is FISInteraction
