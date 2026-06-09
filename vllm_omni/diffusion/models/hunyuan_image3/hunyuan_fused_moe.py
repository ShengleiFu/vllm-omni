# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import vllm.forward_context as _vllm_fc
from vllm.utils.import_utils import resolve_obj_by_qualname

from vllm_omni.platforms import current_omni_platform


def _set_forward_context_num_tokens(num_tokens: int) -> None:
    """Set num_tokens on the vLLM ForwardContext for MoE routing.

    After the rebase to vLLM 0.18.0, FusedMoE expects
    ForwardContext.num_tokens to be set. Without it, MoE expert
    routing may produce incorrect results (silent correctness bug).
    """
    if not _vllm_fc.is_forward_context_available():
        return
    forward_context = _vllm_fc.get_forward_context()
    forward_context.num_tokens = num_tokens
    if not hasattr(forward_context, "in_profile_run"):
        forward_context.in_profile_run = False


class HunyuanFusedMoEDefault:
    """Wrapper around the upstream MoERunner for HunyuanImage3.

    Upstream commit dc68bd8c41 refactored FusedMoE from a class (``nn.Module``)
    to a factory function that returns a ``MoERunner`` instance.  This wrapper
    adapts the old subclass interface to the new factory API while preserving
    the omni-specific forward-context setup and kernel-initialisation hook.
    """

    def __init__(self, *, prefix: str = "", **kwargs: Any) -> None:
        # Current vLLM FusedMoE handles output reduction internally.
        kwargs.pop("reduce_results", None)
        # FusedMoE is now a factory function — call it to get a MoERunner.
        from vllm.model_executor.layers.fused_moe import FusedMoE as _FusedMoE

        self._moe_runner = _FusedMoE(prefix=prefix, **kwargs)
        self._prefix = prefix
        # Install the kernel-init hook on the inner MoERunner (which is a
        # proper nn.Module).
        self._init_hook_handle = self._moe_runner.register_forward_pre_hook(
            self._initialize_kernel_hook, with_kwargs=True
        )

    def _initialize_kernel_hook(self, module: Any, args: Any, kwargs: Any) -> None:
        if (
            hasattr(self._moe_runner, "quant_method")
            and self._moe_runner.quant_method is not None
            and getattr(self._moe_runner.quant_method, "moe_kernel", None) is None
        ):
            self._moe_runner.quant_method.process_weights_after_loading(self._moe_runner)
        self._init_hook_handle.remove()

    @staticmethod
    def make_expert_params_mapping(
        model: Any,
        ckpt_gate_proj_name: str,
        ckpt_down_proj_name: str,
        ckpt_up_proj_name: str,
        num_experts: int,
        num_redundant_experts: int = 0,
    ) -> list[tuple[str, str, int, str]]:
        """Delegate to the upstream standalone function.

        Upstream vLLM refactored ``FusedMoE`` from a class (which had
        ``make_expert_params_mapping`` as a classmethod) to a factory
        function.  The method was moved to a standalone function
        ``fused_moe_make_expert_params_mapping`` in
        ``vllm.model_executor.layers.fused_moe``.
        """
        from vllm.model_executor.layers.fused_moe import (
            fused_moe_make_expert_params_mapping,
        )

        return fused_moe_make_expert_params_mapping(
            model,
            ckpt_gate_proj_name=ckpt_gate_proj_name,
            ckpt_down_proj_name=ckpt_down_proj_name,
            ckpt_up_proj_name=ckpt_up_proj_name,
            num_experts=num_experts,
            num_redundant_experts=num_redundant_experts,
        )

    def forward(self, hidden_states: Any, router_logits: Any) -> Any:
        _set_forward_context_num_tokens(hidden_states.shape[0])
        return self._moe_runner(hidden_states=hidden_states, router_logits=router_logits)


class HunyuanFusedMoE:
    def __new__(cls, *, prefix: str = "", **kwargs: Any) -> Any:
        op_name = "hunyuan_fused_moe"
        current_omni_platform.prepare_diffusion_op_runtime(op_name)
        impl = resolve_obj_by_qualname(
            current_omni_platform.get_diffusion_model_impl_qualname(op_name),
        )
        return impl(prefix=prefix, **kwargs)

    @classmethod
    def make_expert_params_mapping(
        cls,
        model: Any,
        ckpt_gate_proj_name: str,
        ckpt_down_proj_name: str,
        ckpt_up_proj_name: str,
        num_experts: int,
        num_redundant_experts: int = 0,
    ) -> list[tuple[str, str, int, str]]:
        impl = resolve_obj_by_qualname(
            current_omni_platform.get_diffusion_model_impl_qualname("hunyuan_fused_moe"),
        )
        return impl.make_expert_params_mapping(
            model,
            ckpt_gate_proj_name=ckpt_gate_proj_name,
            ckpt_down_proj_name=ckpt_down_proj_name,
            ckpt_up_proj_name=ckpt_up_proj_name,
            num_experts=num_experts,
            num_redundant_experts=num_redundant_experts,
        )
