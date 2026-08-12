# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
import time

import pytest

from tests.helpers.deadline import build_with_cleanup, run_with_deadline

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_completed_work_reports_success():
    done = []
    assert run_with_deadline(lambda: done.append(True), timeout=5.0, label="fast", report=lambda _: None)
    assert done == [True]


def test_stuck_work_returns_instead_of_blocking():
    release = threading.Event()
    messages: list[str] = []

    started = time.monotonic()
    finished = run_with_deadline(release.wait, timeout=0.2, label="stuck", report=messages.append)
    elapsed = time.monotonic() - started

    # The point of the helper: a teardown that never returns must not hold the caller.
    assert finished is False
    assert elapsed < 3.0, elapsed
    assert "did not finish within" in messages[0]
    release.set()


def test_raising_work_is_reported_not_propagated():
    messages: list[str] = []

    # A teardown failure must not replace the error that caused the teardown.
    assert run_with_deadline(
        lambda: (_ for _ in ()).throw(RuntimeError("boom")),
        timeout=5.0,
        label="raising",
        report=messages.append,
    )
    assert "boom" in messages[0]


def test_failed_build_runs_cleanup_and_reraises():
    cleaned = []

    def factory():
        raise OSError("Can't load image processor")

    with pytest.raises(OSError, match="image processor"):
        build_with_cleanup(factory, lambda: cleaned.append(True))

    # Without this the constructor's half-built engine subprocesses are abandoned.
    assert cleaned == [True]


def test_successful_build_skips_cleanup():
    cleaned = []
    assert build_with_cleanup(lambda: "runner", lambda: cleaned.append(True)) == "runner"
    assert cleaned == []


def test_cleanup_failure_does_not_mask_the_build_error():
    messages: list[str] = []

    def cleanup():
        raise RuntimeError("cleanup also broke")

    # The build error is what the test author needs; a cleanup error must not replace it.
    with pytest.raises(OSError, match="original"):
        build_with_cleanup(lambda: (_ for _ in ()).throw(OSError("original")), cleanup, report=messages.append)

    assert "cleanup also broke" in messages[0]
