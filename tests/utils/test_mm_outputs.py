import torch

from vllm_omni.utils import mm_outputs as mm_outputs_mod
from vllm_omni.utils.mm_outputs import build_mm_cpu


def test_build_mm_cpu_converts_once_per_passthrough_key(mocker):
    """Pins the pre-W2 baseline: build_mm_cpu calls _to_cpu exactly once
    per passthrough key for the whole batch, not per request and not
    N x K times. W2's batching must reduce this count; this is the
    "before" measurement the refactor is judged against."""
    spy = mocker.spy(mm_outputs_mod, "_to_cpu")

    multimodal_outputs = {
        "key_a": torch.rand(8, 4),
        "key_b": torch.rand(8, 4),
        "key_c": torch.rand(8, 4),
    }
    result = build_mm_cpu(multimodal_outputs=multimodal_outputs)

    assert spy.call_count == 3
    assert set(result.keys()) == {"key_a", "key_b", "key_c"}


def test_build_mm_cpu_empty_input_is_a_noop(mocker):
    spy = mocker.spy(mm_outputs_mod, "_to_cpu")

    assert build_mm_cpu(multimodal_outputs={}) == {}
    spy.assert_not_called()
