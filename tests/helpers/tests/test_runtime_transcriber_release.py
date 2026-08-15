# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""The server fixture must free the Whisper judge before the next instance starts."""

import threading
from types import SimpleNamespace

import pytest

from tests.helpers import runtime
from tests.helpers.runtime import OmniServerParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def released(monkeypatch) -> list[int]:
    calls: list[int] = []

    class FakeServer:
        def __init__(self, model, serve_args, **kwargs):
            self.model = model

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(runtime, "OmniServer", FakeServer)
    monkeypatch.setattr(runtime, "release_audio_transcriber", lambda: calls.append(1))
    monkeypatch.setattr(
        "tests.helpers.stage_config.stage_config_path_for_run_level",
        lambda path, run_level: None,
    )
    return calls


def _server_generator():
    request = SimpleNamespace(
        param=OmniServerParams(model="fake-model"),
        node=SimpleNamespace(get_closest_marker=lambda name: None),
    )
    return runtime.iter_omni_server(request, "core_model", "", threading.Lock())


def test_server_teardown_releases_the_transcriber(released):
    gen = _server_generator()
    gen.send(None)

    assert released == [], "the worker must stay up while the server is serving"

    with pytest.raises(StopIteration):
        gen.send(None)

    assert released == [1]


def test_failing_test_still_releases_the_transcriber(released):
    """A test that raises must not strand Whisper on the device."""
    gen = _server_generator()
    gen.send(None)

    with pytest.raises(RuntimeError, match="assertion blew up"):
        gen.throw(RuntimeError("assertion blew up"))

    assert released == [1]
