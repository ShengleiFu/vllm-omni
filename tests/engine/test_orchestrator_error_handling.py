# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for error propagation paths within the Orchestrator.

Covers:
- EngineDeadError from an LLM stage poll → fatal error broadcast + replica eviction
- Diffusion stage error output (OmniRequestOutput.from_error) → routed correctly
"""

from __future__ import annotations

import asyncio
import queue
import time
from types import SimpleNamespace

import pytest
from vllm.v1.engine.exceptions import EngineDeadError

from vllm_omni.engine.messages import EngineQueueMessage, ErrorMessage, ShutdownRequestMessage
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput

from .test_orchestrator import (
    FakeOutputProcessor,
    FakeStageClient,
    OrchestratorFixture,
    _build_harness,
    _build_request_output,
    _build_stage_pools,
    _engine_core_outputs,
    _enqueue_add_request,
    _wait_for,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _sampling_params(max_tokens: int = 4):
    from vllm.sampling_params import SamplingParams

    return SamplingParams(max_tokens=max_tokens)


async def _get_any_output_message(fixture: OrchestratorFixture, *, timeout: float = 2.0) -> EngineQueueMessage:
    """Like _get_output_message but returns any message type (including errors)."""
    deadline = time.monotonic() + timeout
    while True:
        if time.monotonic() >= deadline:
            raise AssertionError("Timed out waiting for orchestrator output")
        try:
            return fixture.output_sync_q.get_nowait()
        except queue.Empty:
            await asyncio.sleep(0.01)


@pytest.fixture
def orchestrator_factory():
    fixtures: list[OrchestratorFixture] = []

    def _factory(*args, **kwargs) -> OrchestratorFixture:
        fixture = _build_harness(*args, **kwargs)
        fixtures.append(fixture)
        return fixture

    yield _factory

    for fixture in fixtures:
        if fixture.thread.is_alive():
            fixture.request_sync_q.put_nowait(ShutdownRequestMessage())
            fixture.thread.join(timeout=5)
        for q in fixture.queues:
            q.close()


# ───────── EngineDeadError from LLM stage poll ─────────


class FakeDeadLLMStageClient(FakeStageClient):
    """LLM stage client that raises EngineDeadError on get_output_async."""

    async def get_output_async(self):
        raise EngineDeadError("Stage-0 engine core is dead")


@pytest.mark.asyncio
async def test_engine_dead_error_evicts_replica_and_keeps_running(orchestrator_factory) -> None:
    """When a stage replica raises EngineDeadError during poll, the orchestrator must:
    1. Enqueue a fatal error message for each affected request
    2. Evict the dead replica and keep running so the API server stays up
       (per-replica fault isolation, #4285)
    """
    stage0 = FakeDeadLLMStageClient(stage_type="llm", final_output=True)
    orchestrator_fixture = orchestrator_factory([stage0])
    request = SimpleNamespace(request_id="req-dead", prompt_token_ids=[1, 2])

    try:
        await _enqueue_add_request(
            orchestrator_fixture,
            request_id="req-dead",
            prompt=request,
            original_prompt={"prompt": "hello"},
            sampling_params_list=[_sampling_params()],
            final_stage_id=0,
        )

        # Collect the fatal error message.
        msg = await _get_any_output_message(orchestrator_fixture)

        assert isinstance(msg, ErrorMessage)
        assert msg.type == "error"
        assert msg.fatal is True
        assert msg.request_id == "req-dead"
        assert "Stage-0 engine core is dead" in msg.error

        # The dead replica is evicted; the orchestrator keeps running.
        pool = orchestrator_fixture.orchestrator.stage_pools[0]
        deadline = time.monotonic() + 5.0
        while pool.live_replica_ids() and time.monotonic() < deadline:
            await asyncio.sleep(0.02)
        assert pool.live_replica_ids() == []
        assert orchestrator_fixture.thread.is_alive()

        # Affected request state should be cleaned up.
        assert "req-dead" not in orchestrator_fixture.orchestrator.request_states
    finally:
        if orchestrator_fixture.thread.is_alive():
            orchestrator_fixture.request_sync_q.put_nowait(ShutdownRequestMessage())
            orchestrator_fixture.thread.join(timeout=5)


class _DieOnDemandLLMStageClient(FakeStageClient):
    """LLM stage replica that raises EngineDeadError on poll once ``die`` is set."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.die = False

    async def get_output_async(self):
        if self.die:
            raise EngineDeadError(f"Stage-{self.stage_id} replica-{self.replica_id} is dead")
        return await super().get_output_async()


@pytest.mark.asyncio
async def test_engine_dead_error_fails_only_dead_replica_requests(orchestrator_factory) -> None:
    """Multi-replica stage: one replica dying must fail only the requests bound
    to it. Requests bound to a surviving replica keep running, and the stage
    (still having a live replica) is not torn down (per-replica fault isolation,
    #4285).
    """
    r0 = _DieOnDemandLLMStageClient(stage_type="llm", final_output=True)
    r1 = FakeStageClient(stage_type="llm", final_output=True)
    stage_pools = _build_stage_pools([[r0, r1]])
    orchestrator_fixture = orchestrator_factory([], stage_pools=stage_pools)
    pool = orchestrator_fixture.orchestrator.stage_pools[0]

    try:
        # req-0 → replica 0 (round-robin starts at 0)
        await _enqueue_add_request(
            orchestrator_fixture,
            request_id="req-0",
            prompt=SimpleNamespace(request_id="req-0", prompt_token_ids=[1, 2]),
            original_prompt={"prompt": "a"},
            sampling_params_list=[_sampling_params()],
            final_stage_id=0,
        )
        await _wait_for(lambda: len(r0.add_request_calls) == 1)

        # req-1 → replica 1
        await _enqueue_add_request(
            orchestrator_fixture,
            request_id="req-1",
            prompt=SimpleNamespace(request_id="req-1", prompt_token_ids=[3, 4]),
            original_prompt={"prompt": "b"},
            sampling_params_list=[_sampling_params()],
            final_stage_id=0,
        )
        await _wait_for(lambda: len(r1.add_request_calls) == 1)
        assert pool.get_bound_replica_id("req-0") == 0
        assert pool.get_bound_replica_id("req-1") == 1

        # Kill replica 0 on its next poll.
        r0.die = True

        msg = await _get_any_output_message(orchestrator_fixture)
        assert isinstance(msg, ErrorMessage)
        assert msg.fatal is True
        assert msg.request_id == "req-0"

        # Replica 0 evicted, replica 1 still serving; req-1 untouched.
        await _wait_for(lambda: pool.live_replica_ids() == [1])
        assert orchestrator_fixture.thread.is_alive()
        assert "req-0" not in orchestrator_fixture.orchestrator.request_states
        assert "req-1" in orchestrator_fixture.orchestrator.request_states
    finally:
        if orchestrator_fixture.thread.is_alive():
            orchestrator_fixture.request_sync_q.put_nowait(ShutdownRequestMessage())
            orchestrator_fixture.thread.join(timeout=5)


@pytest.mark.asyncio
async def test_forward_to_dead_downstream_stage_fails_request_not_server(orchestrator_factory) -> None:
    """A request mid-pipeline whose downstream stage has lost all replicas must
    fail that request, not tear down the orchestrator. Forwarding to a stage with
    no live replica is handled gracefully (per-replica fault isolation, #4285).
    """
    stage0 = FakeStageClient(stage_type="llm", final_output=False, next_inputs=[{"prompt_token_ids": [7, 8]}])
    stage1 = _DieOnDemandLLMStageClient(stage_type="llm", final_output=True)
    proc0 = FakeOutputProcessor(request_outputs=[_build_request_output("req-x", token_ids=[3], finished=True)])
    proc1 = FakeOutputProcessor()
    stage_pools = _build_stage_pools([[stage0], [stage1]], output_processors=[proc0, proc1])
    orchestrator_fixture = orchestrator_factory([], stage_pools=stage_pools)
    pool1 = orchestrator_fixture.orchestrator.stage_pools[1]

    try:
        await _enqueue_add_request(
            orchestrator_fixture,
            request_id="req-x",
            prompt=SimpleNamespace(request_id="req-x", prompt_token_ids=[1, 2]),
            original_prompt={"prompt": "hello"},
            sampling_params_list=[_sampling_params(), _sampling_params()],
            final_stage_id=1,
        )
        await _wait_for(lambda: len(stage0.add_request_calls) == 1)

        # Kill stage-1's only replica before stage-0 output is forwarded.
        stage1.die = True
        await _wait_for(lambda: pool1.live_replica_ids() == [])

        # Completing stage 0 triggers a forward to the now-empty stage 1.
        stage0.push_engine_core_outputs(_engine_core_outputs("s0-raw", 1.0))

        # Skip non-error messages (e.g. stage metrics) until the fatal error.
        error_msg: ErrorMessage | None = None
        for _ in range(50):
            m = await _get_any_output_message(orchestrator_fixture)
            if isinstance(m, ErrorMessage):
                error_msg = m
                break
        assert error_msg is not None
        assert error_msg.fatal is True
        assert error_msg.request_id == "req-x"
        assert error_msg.stage_id == 1

        assert orchestrator_fixture.thread.is_alive()
        assert "req-x" not in orchestrator_fixture.orchestrator.request_states
    finally:
        if orchestrator_fixture.thread.is_alive():
            orchestrator_fixture.request_sync_q.put_nowait(ShutdownRequestMessage())
            orchestrator_fixture.thread.join(timeout=5)


@pytest.mark.asyncio
async def test_add_request_to_dead_stage_fails_request_not_server(orchestrator_factory) -> None:
    """A new request entering a stage that already lost all replicas fails that
    request instead of raising out of the request handler and tearing the server
    down. The guard runs before request state is registered, so nothing leaks.
    """
    r0 = _DieOnDemandLLMStageClient(stage_type="llm", final_output=True)
    stage_pools = _build_stage_pools([[r0]])
    orchestrator_fixture = orchestrator_factory([], stage_pools=stage_pools)
    pool = orchestrator_fixture.orchestrator.stage_pools[0]

    try:
        # Send a first request so the replica is polled and dies → evicted.
        await _enqueue_add_request(
            orchestrator_fixture,
            request_id="req-0",
            prompt=SimpleNamespace(request_id="req-0", prompt_token_ids=[1, 2]),
            original_prompt={"prompt": "a"},
            sampling_params_list=[_sampling_params()],
            final_stage_id=0,
        )
        r0.die = True
        await _wait_for(lambda: pool.live_replica_ids() == [])

        # A new request now enters a stage with no live replica.
        await _enqueue_add_request(
            orchestrator_fixture,
            request_id="req-after",
            prompt=SimpleNamespace(request_id="req-after", prompt_token_ids=[3, 4]),
            original_prompt={"prompt": "b"},
            sampling_params_list=[_sampling_params()],
            final_stage_id=0,
        )

        seen_after = False
        for _ in range(50):
            m = await _get_any_output_message(orchestrator_fixture)
            if isinstance(m, ErrorMessage) and m.request_id == "req-after":
                assert m.fatal is True
                assert m.stage_id == 0
                seen_after = True
                break
        assert seen_after

        assert orchestrator_fixture.thread.is_alive()
        # Guard runs before registration, so the failed request never enters state.
        assert "req-after" not in orchestrator_fixture.orchestrator.request_states
    finally:
        if orchestrator_fixture.thread.is_alive():
            orchestrator_fixture.request_sync_q.put_nowait(ShutdownRequestMessage())
            orchestrator_fixture.thread.join(timeout=5)


# ───────── Diffusion stage error output routing ─────────


@pytest.mark.asyncio
async def test_diffusion_error_output_routed_as_finished(orchestrator_factory) -> None:
    """When a diffusion stage returns an OmniRequestOutput with a non-None
    error, the orchestrator must route it as an error message and clean up
    the request state.
    """
    stage0 = FakeStageClient(stage_type="diffusion", final_output=True, final_output_type="image")
    orchestrator_fixture = orchestrator_factory([stage0])
    params = OmniDiffusionSamplingParams()

    try:
        await _enqueue_add_request(
            orchestrator_fixture,
            request_id="req-err",
            prompt={"prompt": "draw a cat"},
            original_prompt={"prompt": "draw a cat"},
            sampling_params_list=[params],
            final_stage_id=0,
        )

        await _wait_for(lambda: len(stage0.add_request_calls) == 1)

        # Push an error output from the diffusion stage.
        stage0.push_diffusion_output(OmniRequestOutput.from_error("req-err", "gpu fault"))

        msg = await _get_any_output_message(orchestrator_fixture)

        assert isinstance(msg, ErrorMessage)
        assert msg.type == "error"
        assert msg.request_id == "req-err"
        assert msg.stage_id == 0
        assert msg.error == "gpu fault"

        # Request state should be cleaned up.
        await _wait_for(lambda: "req-err" not in orchestrator_fixture.orchestrator.request_states)
    finally:
        orchestrator_fixture.request_sync_q.put_nowait(ShutdownRequestMessage())
        orchestrator_fixture.thread.join(timeout=5)


@pytest.mark.asyncio
async def test_diffusion_client_error_output_propagates_status_code(orchestrator_factory) -> None:
    """A client-error OmniRequestOutput (e.g. a guardrail 4xx) from a diffusion
    stage must surface as an ErrorMessage that carries status_code/error_type,
    so the frontend can map it to the correct HTTP response instead of a 500.
    """
    stage0 = FakeStageClient(stage_type="diffusion", final_output=True, final_output_type="image")
    orchestrator_fixture = orchestrator_factory([stage0])
    params = OmniDiffusionSamplingParams()

    try:
        await _enqueue_add_request(
            orchestrator_fixture,
            request_id="req-blocked",
            prompt={"prompt": "blocked"},
            original_prompt={"prompt": "blocked"},
            sampling_params_list=[params],
            final_stage_id=0,
        )

        await _wait_for(lambda: len(stage0.add_request_calls) == 1)

        # Push a client-error output from the diffusion stage.
        stage0.push_diffusion_output(
            OmniRequestOutput.from_error(
                "req-blocked",
                "Input was blocked by Cosmos3 guardrails.",
                status_code=400,
                error_type="BadRequestError",
            )
        )

        msg = await _get_any_output_message(orchestrator_fixture)

        assert isinstance(msg, ErrorMessage)
        assert msg.request_id == "req-blocked"
        assert msg.stage_id == 0
        assert msg.fatal is False
        assert msg.status_code == 400
        assert msg.error_type == "BadRequestError"
        assert msg.error == "Input was blocked by Cosmos3 guardrails."

        # Request state should be cleaned up.
        await _wait_for(lambda: "req-blocked" not in orchestrator_fixture.orchestrator.request_states)
    finally:
        orchestrator_fixture.request_sync_q.put_nowait(ShutdownRequestMessage())
        orchestrator_fixture.thread.join(timeout=5)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status_code", "error_type"),
    [
        pytest.param(429, "RateLimitError", id="429-too-many-requests"),
        pytest.param(413, "PayloadTooLargeError", id="413-payload-too-large"),
        pytest.param(422, "UnprocessableEntityError", id="422-unprocessable-entity"),
        pytest.param(403, "PermissionDeniedError", id="403-forbidden"),
    ],
)
async def test_diffusion_client_error_output_propagates_non_400_status(
    orchestrator_factory,
    status_code: int,
    error_type: str,
) -> None:
    """A diffusion-stage client error with a non-400 4xx status must be serialized
    into an ErrorMessage that preserves that exact status_code/error_type.
    """
    stage0 = FakeStageClient(stage_type="diffusion", final_output=True, final_output_type="image")
    orchestrator_fixture = orchestrator_factory([stage0])
    params = OmniDiffusionSamplingParams()

    try:
        await _enqueue_add_request(
            orchestrator_fixture,
            request_id="req-blocked",
            prompt={"prompt": "blocked"},
            original_prompt={"prompt": "blocked"},
            sampling_params_list=[params],
            final_stage_id=0,
        )

        await _wait_for(lambda: len(stage0.add_request_calls) == 1)

        # Push a non-400 client-error output from the diffusion stage.
        stage0.push_diffusion_output(
            OmniRequestOutput.from_error(
                "req-blocked",
                "client error from diffusion stage",
                status_code=status_code,
                error_type=error_type,
            )
        )

        msg = await _get_any_output_message(orchestrator_fixture)

        assert isinstance(msg, ErrorMessage)
        assert msg.request_id == "req-blocked"
        assert msg.stage_id == 0
        assert msg.fatal is False
        assert msg.status_code == status_code
        assert msg.error_type == error_type
        assert msg.error == "client error from diffusion stage"

        # Request state should be cleaned up.
        await _wait_for(lambda: "req-blocked" not in orchestrator_fixture.orchestrator.request_states)
    finally:
        orchestrator_fixture.request_sync_q.put_nowait(ShutdownRequestMessage())
        orchestrator_fixture.thread.join(timeout=5)
