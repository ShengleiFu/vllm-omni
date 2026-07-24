"""Unit tests for OmniARScheduler._free_request() chunk-transfer-adapter cleanup.

Regression tests for vllm-project/vllm-omni#5349 (P1 follow-up): normal EOS
completion goes through _free_request(), NOT OmniARScheduler.finish_requests()
-- the latter is the external abort/cancel entry point per its own docstring
("Handles the finish signal from outside the scheduler. For example, the API
server can abort a request when the client disconnects."). Without a
chunk_transfer_adapter.cleanup_receiver() call in _free_request(), a request
that finishes normally never leaves self._active_streams: after K requests
complete normally, the bounded-K active-stream window is permanently
exhausted by stale entries and every request after the K-th wedges exactly
like the original #5349 stall -- just delayed to the (K+1)-th request instead
of the 1st.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

# Imports must run in this order: vllm_omni applies patches to vllm.v1.request before
# Request / RequestStatus are bound in this module. Ruff isort would reorder them.
# isort: off
import vllm_omni  # noqa: F401 - import for side effects (patch vLLM)
from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler

# isort: on

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_scheduler(*, chunk_transfer_adapter=None) -> OmniARScheduler:
    """Minimal OmniARScheduler stand-in exercising only _free_request()'s
    "no KV transfer needed" happy path -- the common case for a plain
    text/audio-token generation stage with no PD-disaggregation KV transfer
    configured."""
    sched = OmniARScheduler.__new__(OmniARScheduler)
    sched._omits_kv_transfer_cache = {}
    sched._connector_finished = lambda request: (False, None)
    sched.encoder_cache_manager = MagicMock()
    sched.finished_req_ids = set()
    sched._new_prompt_len_snapshot = {}
    sched.finished_req_ids_dict = None
    sched._should_transfer_kv_for_request = lambda req_id: False
    sched._free_blocks = MagicMock()
    sched._free_input_coordinator_request = MagicMock()
    sched.chunk_transfer_adapter = chunk_transfer_adapter
    return sched


class _FakeFinishedRequest:
    """A minimal finished Request stand-in.

    Deliberately not a SimpleNamespace: SimpleNamespace defines __eq__, which
    makes instances unhashable (hash=None) -- and _free_request() puts the
    request into a set (self._inflight_prefills.discard(request)).
    """

    def __init__(self, request_id: str) -> None:
        self.request_id = request_id

    def is_finished(self) -> bool:
        return True


def _make_finished_request(request_id: str = "req-1"):
    return _FakeFinishedRequest(request_id)


def test_free_request_releases_chunk_transfer_adapter_receiver_state():
    adapter = MagicMock()
    sched = _make_scheduler(chunk_transfer_adapter=adapter)
    request = _make_finished_request("req-1")

    sched._free_request(request)

    adapter.cleanup_receiver.assert_called_once_with("req-1")


def test_free_request_is_safe_without_a_chunk_transfer_adapter():
    """Most stages run without chunk transfer (no downstream connector);
    _free_request() must not assume chunk_transfer_adapter is set."""
    sched = _make_scheduler(chunk_transfer_adapter=None)
    request = _make_finished_request("req-1")

    sched._free_request(request)  # must not raise

    sched._free_input_coordinator_request.assert_called_once_with("req-1")
