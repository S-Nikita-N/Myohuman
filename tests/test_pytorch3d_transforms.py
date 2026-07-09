"""Regression tests for pytorch3d_transforms utilities used by forward_kinematics.

Focus: quat_angle_axis must not mutate its input tensor (previously it
normalized a view of the input in place).
"""
import torch

import myohuman.utils.pytorch3d_transforms as tRot


def _unit_quats(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(n, 4, generator=g)
    return q / q.norm(dim=-1, keepdim=True)


def test_quat_angle_axis_does_not_mutate_input():
    q = _unit_quats(6, seed=1)
    q_before = q.clone()
    angle, axis = tRot.quat_angle_axis(q)
    # input must be untouched
    assert torch.equal(q, q_before)
    # returned axis is unit length
    assert torch.allclose(axis.norm(dim=-1), torch.ones(6), atol=1e-5)
    # angle in [0, pi]
    assert (angle >= 0).all() and (angle <= torch.pi + 1e-5).all()


def test_quat_angle_axis_matches_manual():
    # identity quaternion → angle 0
    q = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    angle, _ = tRot.quat_angle_axis(q)
    assert torch.allclose(angle, torch.zeros(1), atol=1e-6)
