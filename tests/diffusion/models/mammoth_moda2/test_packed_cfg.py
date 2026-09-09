# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CPU tests for CFG packing and request plumbing, without loading model weights.

The fake transformer checks branch conditions and loop semantics. Real-model
attention/RoPE parity and image quality need separate GPU qualification.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from vllm.sampling_params import SamplingParams

from vllm_omni.diffusion.models.mammoth_moda2 import pipeline_mammothmoda2_dit as pipeline_module
from vllm_omni.diffusion.models.mammoth_moda2.pipeline_mammothmoda2_dit import (
    MammothModa2DiTPipeline,
    _pack_cfg_conditions,
)
from vllm_omni.diffusion.utils.param_utils import apply_declared_extra_args
from vllm_omni.model_extras import get_extra_body_params

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.mark.parametrize("positive_length,negative_length", [(3, 0), (0, 3), (2, 5), (5, 2), (0, 0), (3, 3)])
def test_pack_conditions_preserves_branch_order_and_right_padding(positive_length, negative_length):
    positive = torch.arange(positive_length * 2, dtype=torch.bfloat16).reshape(1, positive_length, 2) + 10
    negative = torch.arange(negative_length * 2, dtype=torch.bfloat16).reshape(1, negative_length, 2) - 20
    positive_mask = torch.ones((1, positive_length), dtype=torch.bool)
    negative_mask = torch.ones((1, negative_length), dtype=torch.bool)
    # Preserve existing right padding, including nonzero values under its mask.
    if negative_length > 1:
        negative_mask[:, -1] = False
    before = [tensor.clone() for tensor in (positive, positive_mask, negative, negative_mask)]

    packed, mask = _pack_cfg_conditions(positive, positive_mask, negative, negative_mask)

    assert packed.shape == (2, max(positive_length, negative_length), 2)
    assert packed.dtype == positive.dtype
    assert mask.dtype == torch.bool
    for row, (embeds, original_mask) in enumerate(((positive, positive_mask), (negative, negative_mask))):
        length = embeds.shape[1]
        assert torch.equal(packed[row, :length], embeds[0])
        assert torch.equal(mask[row, :length], original_mask[0])
        assert torch.count_nonzero(packed[row, length:]) == 0
        assert not mask[row, length:].any()
    for actual, original in zip((positive, positive_mask, negative, negative_mask), before):
        assert torch.equal(actual, original)


@pytest.mark.parametrize("invalid_row", [0, 1])
def test_pack_conditions_rejects_mask_holes(invalid_row):
    conditions = [torch.ones(1, 3, 2), torch.ones(1, 3, 2)]
    masks = [torch.ones(1, 3, dtype=torch.bool), torch.ones(1, 3, dtype=torch.bool)]
    masks[invalid_row][0, 1] = False
    with pytest.raises(ValueError, match="contiguous valid prefix"):
        _pack_cfg_conditions(conditions[0], masks[0], conditions[1], masks[1])


def test_pack_conditions_rejects_multiple_requests():
    with pytest.raises(ValueError, match="single-request"):
        _pack_cfg_conditions(
            torch.ones(2, 3, 2),
            torch.ones(2, 3, dtype=torch.bool),
            torch.empty(1, 0, 2),
            torch.empty(1, 0, dtype=torch.bool),
        )


class _Transformer(nn.Module):
    def __init__(self, *, nested=False):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(in_channels=1)
        self.time_caption_embed = SimpleNamespace(image_embedder=object() if nested else None)
        self.calls = []

    def forward(self, **kwargs):
        self.calls.append(
            {key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in kwargs.items()}
        )
        hidden = kwargs["hidden_states"]
        context = kwargs["text_hidden_states"]
        mask = kwargs["text_attention_mask"]
        conditioning = (context * mask.unsqueeze(-1)).sum(dim=(1, 2))
        if kwargs.get("ar_image_hidden_states") is not None:
            conditioning = conditioning + kwargs["ar_image_hidden_states"].sum(dim=(1, 2))
        return hidden * 0.125 + conditioning[:, None, None, None] + kwargs["timestep"][:, None, None, None]


class _ImageRefiner(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.calls = 0

    def forward(self, image, padding_mask):
        self.calls += 1
        assert not padding_mask.any()
        # Make the refined output distinguishable from raw AR image conditions.
        return image + 100


class _Scheduler:
    def __init__(self):
        self.calls = []

    def set_timesteps(self, *, num_inference_steps, device, num_tokens):
        self.timesteps = torch.arange(num_inference_steps, device=device, dtype=torch.float32)

    def step(self, prediction, timestep, latents, *, return_dict):
        assert prediction.shape == latents.shape
        assert latents.shape[0] == 1
        self.calls.append((prediction.clone(), latents.clone()))
        return (latents - prediction * 0.125,)


class _VAE(nn.Module):
    config = SimpleNamespace(scaling_factor=None, shift_factor=None)

    def decode(self, latents, *, return_dict):
        return (latents,)


@pytest.fixture
def pipeline_factory(monkeypatch):
    def build(*, refined=False, model_type="mammothmoda2_qwen2_5_vl", nested=False):
        pipeline = MammothModa2DiTPipeline.__new__(MammothModa2DiTPipeline)
        nn.Module.__init__(pipeline)
        pipeline.config = SimpleNamespace(llm_config=SimpleNamespace(model_type=model_type))
        pipeline.gen_transformer = _Transformer(nested=nested)
        pipeline.gen_image_condition_refiner = _ImageRefiner() if refined else None
        pipeline.gen_vae = _VAE()
        pipeline.gen_freqs_cis = object()
        scheduler = _Scheduler()
        monkeypatch.setattr(pipeline_module, "FlowMatchEulerDiscreteScheduler", lambda: scheduler)
        monkeypatch.setattr(pipeline_module, "randn_tensor", lambda shape, **kwargs: torch.zeros(shape, **kwargs))
        return pipeline, scheduler

    return build


def _runtime_info(*, guidance=4.0, cfg_range=(0.0, 1.0), negative=None, negative_mask=None):
    return {
        "text_prompt_embeds": torch.tensor([[2.0, 3.0], [5.0, 7.0]]),
        "image_prompt_embeds": torch.tensor([[11.0, 13.0]]),
        "negative_prompt_embeds": negative,
        "negative_prompt_attention_mask": negative_mask,
        "image_height": [16],
        "image_width": [16],
        "text_guidance_scale": [guidance],
        "cfg_range": list(cfg_range),
        "num_inference_steps": [4],
    }


@pytest.mark.parametrize(
    "guidance,cfg_range,packed_batch_sizes,sequential_calls",
    [
        (4.0, (0.0, 1.0), [2, 2, 2, 2], 8),
        (4.0, (0.25, 0.5), [1, 2, 2, 1], 6),
        (4.0, (0.0, 0.0), [2, 1, 1, 1], 5),
        (4.0, (0.5, 0.5), [1, 1, 2, 1], 5),
        # i / N never reaches 1, including on the last step.
        (4.0, (1.0, 1.0), [1, 1, 1, 1], 4),
        (1.0, (0.0, 1.0), [1, 1, 1, 1], 4),
        (0.5, (0.0, 1.0), [1, 1, 1, 1], 4),
    ],
)
def test_packed_loop_matches_sequential_and_preserves_cfg_boundaries(
    pipeline_factory, guidance, cfg_range, packed_batch_sizes, sequential_calls
):
    info = _runtime_info(guidance=guidance, cfg_range=cfg_range)
    sequential, sequential_scheduler = pipeline_factory()
    reference = sequential(runtime_additional_information=[info]).multimodal_outputs
    packed, packed_scheduler = pipeline_factory()
    result = packed(
        runtime_additional_information=[info],
        sampling_extra_args=[{"cfg_execution_mode": "packed"}],
    ).multimodal_outputs

    assert len(sequential.gen_transformer.calls) == sequential_calls
    assert all(call["hidden_states"].shape[0] == 1 for call in sequential.gen_transformer.calls)
    assert [call["hidden_states"].shape[0] for call in packed.gen_transformer.calls] == packed_batch_sizes
    assert len(sequential_scheduler.calls) == len(packed_scheduler.calls) == 4
    for reference_step, packed_step in zip(sequential_scheduler.calls, packed_scheduler.calls):
        torch.testing.assert_close(packed_step[0], reference_step[0], rtol=0, atol=0)
        torch.testing.assert_close(packed_step[1], reference_step[1], rtol=0, atol=0)
    torch.testing.assert_close(result, reference, rtol=0, atol=0)
    for call in packed.gen_transformer.calls:
        if call["hidden_states"].shape[0] == 2:
            assert torch.equal(call["hidden_states"][0], call["hidden_states"][1])
            assert torch.equal(call["timestep"][0], call["timestep"][1])


@pytest.mark.parametrize("negative_length", [0, 1, 5])
def test_packed_loop_preserves_refined_positive_and_negative_conditions(pipeline_factory, negative_length):
    negative = torch.arange(negative_length * 2, dtype=torch.float32).reshape(negative_length, 2) - 10
    # A nonempty physical condition can also represent a fully masked null row.
    negative_mask = [index < max(0, negative_length - 1) for index in range(negative_length)]
    info = _runtime_info(negative=negative, negative_mask=negative_mask)
    sequential, _ = pipeline_factory(refined=True)
    reference = sequential(runtime_additional_information=[info]).multimodal_outputs
    pipeline, _ = pipeline_factory(refined=True)
    result = pipeline(
        runtime_additional_information=[info], sampling_extra_args=[{"cfg_execution_mode": "packed"}]
    ).multimodal_outputs

    torch.testing.assert_close(result, reference, rtol=0, atol=0)
    assert pipeline.gen_image_condition_refiner.calls == 1
    expected_positive = torch.cat([info["text_prompt_embeds"], info["image_prompt_embeds"] + 100], dim=0)
    for call in pipeline.gen_transformer.calls:
        assert call.get("ar_image_hidden_states") is None
        assert call.get("ar_image_attention_mask") is None
        context, mask = call["text_hidden_states"], call["text_attention_mask"]
        assert torch.equal(context[0, :3], expected_positive)
        assert torch.equal(context[1, :negative_length], negative)
        assert mask[0, :3].all() and not mask[0, 3:].any()
        assert torch.equal(mask[1, :negative_length], torch.tensor(negative_mask, dtype=torch.bool))
        assert not mask[1, negative_length:].any()


def test_packed_static_conditions_are_built_once(pipeline_factory, monkeypatch):
    calls = []

    def track_pack(*args):
        calls.append(args)
        return _pack_cfg_conditions(*args)

    monkeypatch.setattr(pipeline_module, "_pack_cfg_conditions", track_pack)
    pipeline, _ = pipeline_factory()
    pipeline(runtime_additional_information=[_runtime_info()], sampling_extra_args=[{"cfg_execution_mode": "packed"}])
    assert len(calls) == 1
    conditions = [call["text_hidden_states"] for call in pipeline.gen_transformer.calls]
    assert all(torch.equal(condition, conditions[0]) for condition in conditions)


@pytest.mark.parametrize(
    "guidance,cfg_range",
    [(4.0, (1.0, 1.0)), (4.0, (0.1, 0.2)), (1.0, (0.0, 1.0)), (0.5, (0.0, 1.0))],
)
def test_no_packing_when_no_step_uses_cfg(pipeline_factory, monkeypatch, guidance, cfg_range):
    def unexpected_pack(*args):
        pytest.fail("Conditions must not be packed when no denoising step uses CFG")

    monkeypatch.setattr(pipeline_module, "_pack_cfg_conditions", unexpected_pack)
    # This negative condition is never consumed: the request is conditional-only.
    # Its mask must not be subjected to packed-only validation either.
    info = _runtime_info(
        guidance=guidance,
        cfg_range=cfg_range,
        negative=torch.ones(3, 2),
        negative_mask=[True, False, True],
    )
    sequential, _ = pipeline_factory()
    reference = sequential(runtime_additional_information=[info]).multimodal_outputs
    packed, scheduler = pipeline_factory()
    result = packed(
        runtime_additional_information=[info], sampling_extra_args=[{"cfg_execution_mode": "packed"}]
    ).multimodal_outputs
    assert len(packed.gen_transformer.calls) == len(scheduler.calls) == 4
    assert all(call["hidden_states"].shape[0] == 1 for call in packed.gen_transformer.calls)
    torch.testing.assert_close(result, reference, rtol=0, atol=0)


def test_extra_body_mode_reaches_forward_and_overrides_legacy_sampling_knobs(pipeline_factory):
    params = SamplingParams()
    apply_declared_extra_args(
        params,
        get_extra_body_params("MammothModa2ForConditionalGeneration"),
        {"cfg_execution_mode": "packed", "text_guidance_scale": 4.0, "cfg_range": [0.0, 0.0]},
    )
    pipeline, scheduler = pipeline_factory()
    pipeline(
        runtime_additional_information=[_runtime_info(guidance=1.0)],
        sampling_extra_args=[params.extra_args],
    )
    assert [call["hidden_states"].shape[0] for call in pipeline.gen_transformer.calls] == [2, 1, 1, 1]
    # Positive sum = 41, null prediction = 0, guidance = 4, first latent/timestep = 0.
    assert torch.equal(scheduler.calls[0][0], torch.full_like(scheduler.calls[0][0], 164.0))


@pytest.mark.parametrize("mode", ["typo", True, 2])
def test_invalid_execution_mode_fails_before_transformer(pipeline_factory, mode):
    pipeline, _ = pipeline_factory()
    with pytest.raises(ValueError, match="cfg_execution_mode"):
        pipeline(runtime_additional_information=[_runtime_info()], sampling_extra_args=[{"cfg_execution_mode": mode}])
    assert pipeline.gen_transformer.calls == []


@pytest.mark.parametrize(
    "model_type,nested",
    [("mammothmoda2_qwen3_vl", True), ("mammothmoda2_qwen3_vl", False), ("mammothmoda2_qwen2_5_vl", True)],
)
def test_packed_rejects_unqualified_model_variants(pipeline_factory, model_type, nested):
    pipeline, _ = pipeline_factory(model_type=model_type, nested=nested)
    with pytest.raises(NotImplementedError, match="Preview only"):
        pipeline(
            runtime_additional_information=[_runtime_info()], sampling_extra_args=[{"cfg_execution_mode": "packed"}]
        )
    assert pipeline.gen_transformer.calls == []


def test_dev_default_sequential_keeps_image_conditioning_on_positive_only(pipeline_factory):
    pipeline, _ = pipeline_factory(model_type="mammothmoda2_qwen3_vl", nested=True)
    pipeline(runtime_additional_information=[_runtime_info()])
    assert len(pipeline.gen_transformer.calls) == 8
    for index, call in enumerate(pipeline.gen_transformer.calls):
        assert call["hidden_states"].shape[0] == 1
        if index % 2 == 0:
            assert call["ar_image_hidden_states"] is not None
        else:
            assert call.get("ar_image_hidden_states") is None
