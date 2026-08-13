# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Startup-failure teardown must be bounded, or the startup error never surfaces.

When stage initialization fails, the caller is holding the exception that a test (or a
serving client) needs to see. Everything on the way out has to finish within a budget:
an unbounded wait there means the error is never reported at all and the run dies on an
outer timeout instead.
"""

import concurrent.futures
import multiprocessing
import threading
import time
import types

import pytest

from vllm_omni.engine import stage_runtime as stage_runtime_module
from vllm_omni.engine.stage_runtime import StageRuntime

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

# Generous enough that a slow CI agent will not flake, tight enough that an unbounded
# wait (the bug) cannot pass: the wedged worker below never finishes on its own.
_RETURN_BUDGET_S = 30.0


def _sleep_forever() -> None:
    time.sleep(3600)


def _bare_runtime(stage_pools, executor=None) -> StageRuntime:
    """A StageRuntime without __init__: initialization is what has just failed."""
    runtime = object.__new__(StageRuntime)
    runtime.stage_pools = stage_pools
    runtime._stage_init_executor = executor
    return runtime


def _pool(*clients):
    return types.SimpleNamespace(clients=list(clients))


def _spawn_child() -> multiprocessing.Process:
    proc = multiprocessing.Process(target=_sleep_forever, name="fake-stage-proc", daemon=True)
    proc.start()
    return proc


def test_owned_processes_finds_both_manager_shapes():
    # Diffusion clients hold a single `.proc`; engine-core clients hold `.processes`.
    diffusion_proc = types.SimpleNamespace(name="StageDiffusionProc")
    core_procs = [types.SimpleNamespace(name="EngineCore_0"), types.SimpleNamespace(name="EngineCore_1")]
    diffusion_client = types.SimpleNamespace(_proc_manager=types.SimpleNamespace(proc=diffusion_proc))
    core_client = types.SimpleNamespace(
        resources=types.SimpleNamespace(engine_manager=types.SimpleNamespace(processes=core_procs))
    )
    runtime = _bare_runtime([_pool(diffusion_client, core_client, None)])

    owned = runtime.owned_processes()

    assert diffusion_proc in owned
    assert all(proc in owned for proc in core_procs)
    assert len(owned) == 3


def test_startup_failure_shutdown_terminates_owned_processes():
    procs = [_spawn_child(), _spawn_child()]
    client = types.SimpleNamespace(
        resources=types.SimpleNamespace(engine_manager=types.SimpleNamespace(processes=procs))
    )
    runtime = _bare_runtime([_pool(client)])

    try:
        runtime.shutdown_after_startup_failure(timeout=_RETURN_BUDGET_S)
        # Without this the children outlive the failed init and get joined at
        # interpreter exit, which is the stall this guards against.
        for proc in procs:
            assert not proc.is_alive(), f"{proc.name} survived startup-failure teardown"
    finally:
        for proc in procs:
            if proc.is_alive():
                proc.kill()
                proc.join(5)


def test_returns_even_while_a_stage_init_worker_is_wedged():
    release = threading.Event()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    wedged = executor.submit(release.wait)
    runtime = _bare_runtime([], executor=executor)

    started = time.monotonic()
    try:
        runtime.shutdown_after_startup_failure(timeout=_RETURN_BUDGET_S)
        elapsed = time.monotonic() - started

        # shutdown(wait=True) on the executor would block here until the worker returns,
        # which it never does -- that is the unbounded wait on the startup-failure path.
        assert elapsed < _RETURN_BUDGET_S, elapsed
        assert runtime._stage_init_executor is None
    finally:
        release.set()
        wedged.result(timeout=10)
        executor.shutdown(wait=True)


def test_process_shutdown_failure_is_logged_not_raised(monkeypatch):
    def _explode(procs, timeout=None):
        raise RuntimeError("terminate failed")

    monkeypatch.setattr(stage_runtime_module, "shutdown_procs", _explode)
    client = types.SimpleNamespace(
        resources=types.SimpleNamespace(engine_manager=types.SimpleNamespace(processes=[types.SimpleNamespace()]))
    )
    runtime = _bare_runtime([_pool(client)])

    # The startup exception is already propagating; a teardown failure must not replace it.
    runtime.shutdown_after_startup_failure(timeout=_RETURN_BUDGET_S)


def test_unreadable_client_does_not_raise():
    class Hostile:
        @property
        def resources(self):
            raise RuntimeError("client is half-built")

    runtime = _bare_runtime([_pool(Hostile())])

    runtime.shutdown_after_startup_failure(timeout=_RETURN_BUDGET_S)
