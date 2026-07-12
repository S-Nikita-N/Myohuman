"""Stage-2 tests: require a real GPU/MPS accelerator (or RUN_STAGE2=1).

Skipped by default (stage-1 is CPU-only). They assert the torch transform
math is device-agnostic: the same inputs give the same result on the
accelerator as on CPU, within float tolerance. No data files needed.
"""

import torch
import pytest

import myohuman.utils.pytorch3d_transforms as tRot

pytestmark = pytest.mark.stage2


########################################
#            Device parity             #
########################################


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    pytest.skip("no GPU/MPS accelerator available")


def _unit_quats(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(n, 4, generator=g)
    return q / q.norm(dim=-1, keepdim=True)


def test_quaternion_to_matrix_matches_cpu():
    q = _unit_quats(8, seed=1)
    cpu = tRot.quaternion_to_matrix(q)

    dev = _device()
    got = tRot.quaternion_to_matrix(q.to(dev))

    assert got.device.type == dev.type
    torch.testing.assert_close(got.cpu(), cpu, rtol=1e-5, atol=1e-6)


def test_matrix_to_quaternion_roundtrip_on_device():
    q = _unit_quats(8, seed=2)
    dev = _device()
    mat = tRot.quaternion_to_matrix(q.to(dev))
    q_back = tRot.matrix_to_quaternion(mat)

    # quaternion sign ambiguity: compare |q| to |q_back|
    torch.testing.assert_close(
        q_back.cpu().abs(),
        q.abs(),
        rtol=1e-4,
        atol=1e-5,
    )
