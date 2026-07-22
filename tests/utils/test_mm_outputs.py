import pytest
import torch

from vllm_omni.utils import mm_outputs as mm_outputs_mod
from vllm_omni.utils.mm_outputs import build_mm_cpu

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_build_mm_cpu_converts_once_per_passthrough_key(mocker):
    """Pins the pre-W2 baseline: build_mm_cpu invokes the per-tensor
    conversion helper (_to_cpu) exactly once per passthrough key for the
    whole batch, not per request and not N x K times. This test runs on
    CPU tensors, so it characterizes the call-count baseline, not an
    actual measured device-to-host transfer; W2's batching must reduce
    this count, and this is the "before" measurement the refactor is
    judged against."""
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


def test_build_mm_cpu_preserves_nested_and_non_tensor_payload_semantics():
    """_to_cpu must recurse into nested dict/list structures, force
    non-contiguous tensors to become contiguous, detach from autograd,
    and leave dtype/shape/values unchanged; non-tensor leaves (scalar,
    string) pass through untouched. Any W2 implementation (cat, deferral,
    or a new materializer) must preserve this exact semantics."""
    non_contig = torch.arange(9, dtype=torch.float32).reshape(3, 3).t()
    assert not non_contig.is_contiguous()
    grad_tensor = torch.tensor([5.0, 6.0], requires_grad=True)

    payload = {
        "nested": {"inner_tensor": non_contig, "inner_scalar": "keep-me"},
        "list_of_tensors": [torch.tensor([1.0, 2.0]), "marker", torch.tensor([3.0, 4.0])],
        "grad_key": grad_tensor,
        "scalar": 42,
    }
    result = build_mm_cpu(multimodal_outputs=payload)

    inner = result["nested"]["inner_tensor"]
    assert inner.is_contiguous()
    assert inner.dtype == non_contig.dtype
    assert inner.shape == non_contig.shape
    assert torch.equal(inner, non_contig)
    assert result["nested"]["inner_scalar"] == "keep-me"

    lst = result["list_of_tensors"]
    assert torch.equal(lst[0], torch.tensor([1.0, 2.0]))
    assert lst[1] == "marker"
    assert torch.equal(lst[2], torch.tensor([3.0, 4.0]))

    assert result["grad_key"].requires_grad is False
    assert torch.equal(result["grad_key"], grad_tensor.detach())

    assert result["scalar"] == 42
