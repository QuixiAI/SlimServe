"""A platform record may opt into speculation on its own (glm53-nvfp4-4)."""

from slimserve import registry
from slimserve.engine import _speculative_config


def _plan(platform):
    return registry.resolve("glm53-nvfp4-4", platform, 4, None, 0)


def test_rtx6000_record_opts_into_mtp_while_a100_stays_off():
    rtx = _plan("rtx6000")
    a100 = _plan("a100")
    assert rtx.speculative is True
    assert a100.speculative is False
    spec = _speculative_config(rtx)
    assert spec["method"] == "mtp"
    assert spec["num_speculative_tokens"] == 1  # the k=1 equivalence gate first
    assert spec["moe_backend"] == "triton"
    assert spec["attention_backend"] == "QUIXICORE_MLA_SPARSE"
    assert spec["index_share_for_mtp_iteration"] is True
    assert _speculative_config(a100) is None
