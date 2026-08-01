import torch

from daph_ext.mixers.gdn2_reference import gdn2_recurrence_reference


def test_zero_write_zero_state_stays_zero():
    k = torch.randn(2, 5, 3, 4) * 0.1
    v = torch.randn(2, 5, 3, 6) * 0.1
    erase = torch.sigmoid(torch.randn_like(k))
    write = torch.zeros_like(v)
    log_decay = -torch.ones_like(k) * 0.1
    y, state = gdn2_recurrence_reference(k, v, erase, write, log_decay)
    assert torch.equal(y, torch.zeros_like(y))
    assert torch.equal(state, torch.zeros_like(state))


def test_shapes_and_finite():
    b, t, h, kd, vd = 1, 7, 2, 3, 5
    k = torch.randn(b, t, h, kd) * 0.05
    v = torch.randn(b, t, h, vd) * 0.05
    erase = torch.sigmoid(torch.randn_like(k))
    write = torch.sigmoid(torch.randn_like(v))
    log_decay = -torch.nn.functional.softplus(torch.randn_like(k)) * 0.05
    y, state = gdn2_recurrence_reference(k, v, erase, write, log_decay)
    assert y.shape == (b, t, h, vd)
    assert state.shape == (b, h, kd, vd)
    assert torch.isfinite(y).all()
    assert torch.isfinite(state).all()
