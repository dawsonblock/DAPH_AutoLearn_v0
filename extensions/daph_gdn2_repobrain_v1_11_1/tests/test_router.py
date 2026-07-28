import pytest

from daph_ext.config import MixerKind
from daph_ext.mixers.router import alternating_gdn2_attention, assert_no_sparse_compaction


def test_alternating_plan():
    plan = alternating_gdn2_attention(6, 2)
    assert plan.kinds == (
        MixerKind.GDN2,
        MixerKind.ATTENTION,
        MixerKind.GDN2,
        MixerKind.ATTENTION,
        MixerKind.GDN2,
        MixerKind.ATTENTION,
    )


def test_sparse_compaction_guard():
    with pytest.raises(RuntimeError):
        assert_no_sparse_compaction([0, 1, 2, 3, 4], [0, 3, 4])
