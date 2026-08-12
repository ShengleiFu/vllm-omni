# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Bounded setup and teardown for fixtures that own a live engine.

Tearing an engine down can block indefinitely: ``destroy_process_group()`` waits on
peer ranks that a mid-initialization failure has already killed, and a stuck
``close()`` carries no timeout of its own. A fixture that blocks there never returns
control to pytest, so the run reports no failure at all and dies on the CI step
timeout instead -- which hides the error that started it.

Deliberately free of vllm imports so these can be unit tested without an accelerator.
"""

import threading
from collections.abc import Callable
from typing import TypeVar

_T = TypeVar("_T")


def run_with_deadline(
    fn: Callable[[], None],
    timeout: float,
    label: str,
    report: Callable[[str], None] = print,
) -> bool:
    """Run ``fn`` on a daemon thread, giving up the wait after ``timeout`` seconds.

    Returns True when ``fn`` finished. On overrun the thread is left running: it is a
    daemon, so it cannot hold the interpreter open, and the caller continues, because
    the alternative is blocking forever. Exceptions are reported rather than raised --
    a teardown failure must not replace the error that caused the teardown.
    """
    raised: list[BaseException] = []

    def _run() -> None:
        try:
            fn()
        except BaseException as exc:  # noqa: BLE001 - must not mask the original error
            raised.append(exc)

    thread = threading.Thread(target=_run, name=f"deadline-{label}", daemon=True)
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        report(f"{label} did not finish within {timeout:g}s; continuing without it")
        return False
    if raised:
        report(f"{label} raised {raised[0]!r}; continuing")
    return True


def build_with_cleanup(
    factory: Callable[[], _T],
    cleanup: Callable[[], None],
    report: Callable[[str], None] = print,
) -> _T:
    """Build via ``factory``, running ``cleanup`` if it raises, and re-raise regardless.

    ``with Runner(...) as runner`` evaluates the constructor before ``__enter__``, so a
    constructor that fails part-way never reaches ``__exit__`` and abandons whatever it
    already started -- engine subprocesses among it, which the interpreter then joins at
    exit. Callers that own that cleanup have to run it here instead.

    A failure inside ``cleanup`` is reported and dropped: the build error is the one the
    test author needs, and letting the cleanup error replace it hides the cause.
    """
    try:
        return factory()
    except BaseException:
        try:
            cleanup()
        except BaseException as exc:  # noqa: BLE001 - must not mask the build error
            report(f"cleanup after a failed build raised {exc!r}; continuing")
        raise
