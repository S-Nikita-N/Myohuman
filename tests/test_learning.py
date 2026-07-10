"""Characterization tests for the CPU learning modules used at inference.

MLP / RunningNorm / policies — no data files, no GPU. torch is seeded so
weight init is reproducible. Only the well-defined default code paths are
pinned here (e.g. RunningNorm with its default decay=1, where the update
reduces to a standard running mean/variance).
"""
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from myohuman.learning.mlp import MLP
from myohuman.learning.running_norm import RunningNorm
from myohuman.learning.policy_gaussian import PolicyGaussian, DiagGaussian
from myohuman.learning.policy_lattice import PolicyLattice


def _cfg(units=(32, 32), activation="silu", fix_std=True, log_std=-2.0):
    return OmegaConf.create(
        {"learning": {"mlp": {"units": list(units), "activation": activation},
                      "fix_std": fix_std, "log_std": log_std}}
    )


# ─────────────────────────────────── MLP ───────────────────────────────────
@pytest.mark.parametrize("activation", ["tanh", "relu", "sigmoid", "gelu", "silu"])
def test_mlp_forward_shape(activation):
    torch.manual_seed(0)
    net = MLP(10, (16, 8), activation)
    assert net.out_dim == 8
    out = net(torch.zeros(4, 10))
    assert out.shape == (4, 8)


def test_mlp_unknown_activation_raises():
    # unknown activation must fail loudly at construction, not silently at forward
    with pytest.raises(ValueError):
        MLP(10, (16, 8), "not_an_activation")


# ─────────────────────────────── RunningNorm ───────────────────────────────
def test_running_norm_first_update_is_batch_stats():
    torch.manual_seed(0)
    rn = RunningNorm(3)  # default decay=1
    x = torch.randn(100, 3)
    rn.train()
    rn(x)  # triggers update
    mean = x.mean(0)
    var = x.var(0, unbiased=False)
    assert torch.allclose(rn.mean, mean, atol=1e-6)
    assert torch.allclose(rn.var, var, atol=1e-6)
    assert torch.allclose(rn.std, torch.sqrt(var), atol=1e-6)
    assert int(rn.n) == 100


def test_running_norm_eval_normalizes():
    torch.manual_seed(1)
    rn = RunningNorm(3)
    x = torch.randn(200, 3)
    rn.train()
    rn(x)
    rn.eval()
    y = rn(x)
    expected = torch.clamp((x - rn.mean) / (rn.std + 1e-8), -5.0, 5.0)
    assert torch.allclose(y, expected, atol=1e-6)


def test_running_norm_identity_before_any_update():
    rn = RunningNorm(3)
    rn.eval()  # n == 0 → passthrough
    x = torch.randn(4, 3)
    assert torch.allclose(rn(x), x)


# ─────────────────────────────── DiagGaussian ──────────────────────────────
def test_diag_gaussian_log_prob_matches_normal():
    loc = torch.tensor([[0.5, -1.0, 2.0]])
    scale = torch.tensor([[1.0, 0.5, 2.0]])
    dist = DiagGaussian(loc, scale)
    value = torch.tensor([[0.0, 0.0, 0.0]])
    ref = torch.distributions.Normal(loc, scale).log_prob(value).sum(1, keepdim=True)
    assert torch.allclose(dist.log_prob(value), ref, atol=1e-6)
    assert dist.log_prob(value).shape == (1, 1)


def test_diag_gaussian_kl_zero_against_self_detached():
    loc = torch.tensor([[0.3, 0.7]])
    scale = torch.tensor([[1.0, 2.0]])
    dist = DiagGaussian(loc, scale)
    # KL against detached copy of itself is 0
    assert torch.allclose(dist.kl(), torch.zeros(1, 1), atol=1e-6)


def test_diag_gaussian_mean_sample_is_loc():
    loc = torch.randn(3, 4)
    dist = DiagGaussian(loc, torch.ones(3, 4))
    assert torch.allclose(dist.mean_sample(), loc)


# ─────────────────────────────── PolicyGaussian ────────────────────────────
def test_policy_gaussian_forward():
    torch.manual_seed(7)
    pol = PolicyGaussian(_cfg(fix_std=True, log_std=-1.5), action_dim=6, state_dim=10)
    pol.eval()  # freeze RunningNorm so repeated forwards are deterministic
    dist = pol(torch.randn(5, 10))
    assert isinstance(dist, DiagGaussian)
    assert dist.loc.shape == (5, 6)
    # fixed std → log_std param equals config value everywhere
    assert torch.allclose(pol.action_log_std, torch.full((1, 6), -1.5))
    # mean action == distribution loc for the same input
    x = torch.randn(5, 10)
    a = pol.select_action(x, mean_action=True)
    assert torch.allclose(a, pol(x).loc)


def test_policy_gaussian_action_mean_bias_zeroed():
    torch.manual_seed(3)
    pol = PolicyGaussian(_cfg(), action_dim=4, state_dim=8)
    assert torch.allclose(pol.action_mean.bias, torch.zeros(4))


# ─────────────────────────────── PolicyLattice ─────────────────────────────
def test_policy_lattice_covariance_formula():
    torch.manual_seed(11)
    # latent_dim is tied to the MLP output width (see agent_humanoid.py: latent_dim=512
    # with a 512-wide net); the covariance broadcast requires latent_dim == units[-1].
    units, latent_dim = (16, 16), 16
    action_dim, state_dim = 5, 8
    pol = PolicyLattice(_cfg(units=units, fix_std=True, log_std=-1.0),
                        action_dim, latent_dim, state_dim)
    pol.eval()
    x = torch.randn(2, state_dim)
    dist = pol(x)
    # reconstruct expected covariance: W diag(latent_var) W^T + diag(action_var)
    std = torch.exp(pol.log_std)
    action_var = std[:, :action_dim] ** 2
    latent_var = std[:, action_dim:] ** 2
    W = pol.action_mean.weight
    expected = (W * latent_var[..., None, :]).matmul(W.T)
    idx = torch.arange(action_dim)
    expected[..., idx, idx] += action_var
    assert torch.allclose(dist.covariance_matrix, expected, atol=1e-5)
    assert dist.loc.shape == (2, action_dim)


def test_policy_lattice_mean_action_is_loc():
    torch.manual_seed(13)
    pol = PolicyLattice(_cfg(units=(16, 16)), action_dim=4, latent_dim=16, state_dim=6)
    pol.eval()
    x = torch.randn(3, 6)
    a = pol.select_action(x, mean_action=True)
    assert torch.allclose(a, pol.lattice_dist.loc)


def test_policy_lattice_log_prob_shape():
    torch.manual_seed(17)
    pol = PolicyLattice(_cfg(units=(16, 16)), action_dim=4, latent_dim=16, state_dim=6)
    pol.eval()
    x = torch.randn(3, 6)
    action = torch.randn(3, 4)
    lp = pol.get_log_prob(x, action)
    assert lp.shape == (3, 1)
