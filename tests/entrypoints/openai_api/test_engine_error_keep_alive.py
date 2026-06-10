# tests/entrypoints/openai/test_engine_error_keep_alive.py
"""Single-stage engine death must not terminate the API server (issue #4285).

Omni marks ``errored`` when any stage engine dies, but the orchestrator can
still route requests: subsequent calls fast-fail with ``EngineDeadError`` and
``/health`` reports 503. These tests pin the termination policy: keep uvicorn
alive while the orchestrator is alive, defer to upstream
``terminate_if_errored`` once it is gone.
"""

from types import SimpleNamespace

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from pytest_mock import MockerFixture

from vllm_omni.entrypoints.omni_base import OmniEngineDeadError
from vllm_omni.entrypoints.openai import api_server as api_server_module


def _build_app(engine_client) -> FastAPI:
    app = FastAPI()
    app.state.args = SimpleNamespace(log_error_stack=False)
    app.state.engine_client = engine_client
    app.state.server = SimpleNamespace(should_exit=False)
    api_server_module._register_omni_exception_handlers(app)

    @app.get("/boom")
    async def boom(request: Request):
        request.state.request_metadata = SimpleNamespace(request_id="req-keep-alive-1")
        raise OmniEngineDeadError("engine dead", error_stage_id=1)

    return app


def test_engine_error_keeps_server_alive_when_orchestrator_alive(mocker: MockerFixture):
    terminate_mock = mocker.patch.object(api_server_module, "terminate_if_errored")
    app = _build_app(
        SimpleNamespace(
            engine=SimpleNamespace(is_alive=lambda: True),
            errored=True,
        )
    )

    response = TestClient(app, raise_server_exceptions=False).get("/boom")

    assert response.status_code == 500
    assert response.json()["error"]["error_stage_id"] == 1
    terminate_mock.assert_not_called()
    assert app.state.server.should_exit is False


def test_engine_error_terminates_when_orchestrator_dead(mocker: MockerFixture):
    terminate_mock = mocker.patch.object(api_server_module, "terminate_if_errored")
    app = _build_app(
        SimpleNamespace(
            engine=SimpleNamespace(is_alive=lambda: False),
            errored=True,
        )
    )

    response = TestClient(app, raise_server_exceptions=False).get("/boom")

    assert response.status_code == 500
    terminate_mock.assert_called_once()


def test_engine_error_falls_back_to_upstream_without_orchestrator_liveness(mocker: MockerFixture):
    terminate_mock = mocker.patch.object(api_server_module, "terminate_if_errored")
    app = _build_app(SimpleNamespace(errored=True))

    response = TestClient(app, raise_server_exceptions=False).get("/boom")

    assert response.status_code == 500
    terminate_mock.assert_called_once()
