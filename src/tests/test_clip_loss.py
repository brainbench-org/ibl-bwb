import math

import torch

from core.nn.loss import CLIPLoss

TEMPERATURE = 0.5
NUM_PAIRS = 8


def test_orthonormal_views_match_closed_form():
    # aligned pairs at similarity 1, every negative at 0
    z = torch.eye(NUM_PAIRS)
    expected = math.log(1 + (NUM_PAIRS - 1) * math.exp(-1 / TEMPERATURE))
    loss = CLIPLoss(temperature=TEMPERATURE)(z, z)
    assert torch.allclose(loss, torch.tensor(expected), atol=1e-6)


def test_normalization_is_internal():
    torch.manual_seed(0)
    z_a, z_b = torch.randn(NUM_PAIRS, 16), torch.randn(NUM_PAIRS, 16)
    loss_fn = CLIPLoss(temperature=TEMPERATURE)
    normalized = loss_fn(
        torch.nn.functional.normalize(z_a, dim=-1),
        torch.nn.functional.normalize(z_b, dim=-1),
    )
    assert torch.allclose(loss_fn(z_a * 7.0, z_b * 0.1), normalized, atol=1e-6)


def test_matched_pairs_beat_shuffled_pairs():
    torch.manual_seed(0)
    z_a = torch.randn(NUM_PAIRS, 16)
    z_b = z_a + 0.01 * torch.randn(NUM_PAIRS, 16)
    loss_fn = CLIPLoss(temperature=TEMPERATURE)
    assert loss_fn(z_a, z_b) < loss_fn(z_a, z_b.roll(1, dims=0))


def test_gather_is_noop_outside_ddp():
    torch.manual_seed(0)
    z_a, z_b = torch.randn(NUM_PAIRS, 16), torch.randn(NUM_PAIRS, 16)
    gathered = CLIPLoss(temperature=TEMPERATURE, gather_distributed=True)(z_a, z_b)
    local = CLIPLoss(temperature=TEMPERATURE, gather_distributed=False)(z_a, z_b)
    assert torch.equal(gathered, local)


def test_gradients_reach_both_views():
    torch.manual_seed(0)
    z_a = torch.randn(NUM_PAIRS, 16, requires_grad=True)
    z_b = torch.randn(NUM_PAIRS, 16, requires_grad=True)
    CLIPLoss(temperature=TEMPERATURE)(z_a, z_b).backward()
    assert z_a.grad is not None and z_a.grad.abs().sum() > 0
    assert z_b.grad is not None and z_b.grad.abs().sum() > 0
