# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""No server or runner may initialize while a Whisper worker still holds the device."""

import threading
from types import SimpleNamespace

import pytest

from tests.helpers import runtime
from tests.helpers.runtime import OmniServerParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def events(monkeypatch) -> list[str]:
    """Record construction and release in order, for every backing class."""
    calls: list[str] = []

    class FakeRuntime:
        def __init__(self, *args, **kwargs):
            calls.append("construct")
            self.model = args[0] if args else "fake-model"

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    for name in ("OmniServer", "OmniServerStageCli", "OmniRunner"):
        monkeypatch.setattr(runtime, name, FakeRuntime)
    monkeypatch.setattr(runtime, "release_audio_transcriber", lambda: calls.append("release"))
    monkeypatch.setattr(
        "tests.helpers.stage_config.stage_config_path_for_run_level",
        lambda path, run_level: path,
    )
    return calls


def _node():
    return SimpleNamespace(get_closest_marker=lambda name: None)


def _plain_server():
    request = SimpleNamespace(param=OmniServerParams(model="fake-model"), node=_node())
    return runtime.iter_omni_server(request, "core_model", "", threading.Lock())


def _stage_cli_server():
    request = SimpleNamespace(
        param=OmniServerParams(model="fake-model", stage_config_path="stages.yaml", use_stage_cli=True),
        node=_node(),
    )
    return runtime.iter_omni_server(request, "core_model", "", threading.Lock())


def _runner():
    request = SimpleNamespace(param=("fake-model", None), node=_node())
    return runtime.iter_omni_runner(request, "", "core_model", threading.Lock())


ALL_FIXTURES = pytest.mark.parametrize(
    "make_generator",
    [_plain_server, _stage_cli_server, _runner],
    ids=["omni_server", "omni_server_stage_cli", "omni_runner"],
)


@ALL_FIXTURES
def test_releases_before_construction_and_after_teardown(events, make_generator):
    generator = make_generator()
    generator.send(None)

    assert events == ["release", "construct"], "the device must be clear before the model loads"

    with pytest.raises(StopIteration):
        generator.send(None)

    assert events == ["release", "construct", "release"]


@ALL_FIXTURES
def test_failing_test_still_releases(events, make_generator):
    """A test that raises must not strand Whisper on the device."""
    generator = make_generator()
    generator.send(None)

    with pytest.raises(RuntimeError, match="assertion blew up"):
        generator.throw(RuntimeError("assertion blew up"))

    assert events[-1] == "release"
