# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from tests.helpers import assertions
from tests.helpers.assertions import (
    AUDIO_MISMATCH_MESSAGE,
    assert_omni_response,
    is_audio_mismatch,
    wilson_interval,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_audio_mismatch_is_recognised_with_detail_appended():
    # assert_omni_response appends the transcript to this message, so matching must
    # not require equality.
    exc = AssertionError(f"{AUDIO_MISMATCH_MESSAGE} (short-text containment check failed: text='London')")
    assert is_audio_mismatch(exc)


def test_bare_audio_mismatch_is_recognised():
    assert is_audio_mismatch(AssertionError(AUDIO_MISMATCH_MESSAGE))


@pytest.mark.parametrize(
    "exc",
    [
        AssertionError("The output does not contain any of the keywords."),
        AssertionError("No audio output is generated"),
        AssertionError("The request failed."),
        RuntimeError(AUDIO_MISMATCH_MESSAGE),
        ValueError("boom"),
    ],
)
def test_serving_failures_are_not_absorbed_as_audio_mismatches(exc):
    # A caller tolerating a few mismatches while sampling a success rate must still
    # fail outright on these; misclassifying one would let a broken server pass.
    assert not is_audio_mismatch(exc)


def _fake_response(text: str) -> SimpleNamespace:
    return SimpleNamespace(success=True, text_content=text, audio_content=None, audio_bytes=b"fake-wav")


def _omni_response_raising(monkeypatch, transcript: str) -> AssertionError:
    """Drive the real assert_omni_response to failure and hand back what it raised."""
    monkeypatch.setattr(assertions, "convert_audio_bytes_to_text", lambda *a, **kw: transcript)
    response = _fake_response("London")
    request_config = {"modalities": ["text", "audio"], "key_words": {"text": ["london"]}}
    with pytest.raises(AssertionError) as excinfo:
        assert_omni_response(response, request_config, run_level="full_model")
    return excinfo.value


def test_real_audio_mismatch_failure_is_classified(monkeypatch):
    # Ties the classifier to the actual producer: if assert_omni_response's wording
    # drifts, this fails instead of the success-rate gate silently turning every
    # quality failure into a hard failure.
    assert is_audio_mismatch(_omni_response_raising(monkeypatch, "런던"))


def test_real_keyword_failure_is_not_classified_as_audio_mismatch(monkeypatch):
    monkeypatch.setattr(assertions, "convert_audio_bytes_to_text", lambda *a, **kw: "London")
    response = _fake_response("Paris")
    request_config = {"modalities": ["text", "audio"], "key_words": {"text": ["london"]}}
    with pytest.raises(AssertionError) as excinfo:
        assert_omni_response(response, request_config, run_level="full_model")
    assert not is_audio_mismatch(excinfo.value)


def test_wilson_interval_brackets_the_observed_rate():
    lo, hi = (float(x.rstrip("%")) for x in wilson_interval(29, 40).split("-"))
    assert lo < 72.5 < hi


@pytest.mark.parametrize(("successes", "total"), [(40, 40), (0, 40)])
def test_wilson_interval_stays_within_bounds_at_the_extremes(successes, total):
    lo, hi = (float(x.rstrip("%")) for x in wilson_interval(successes, total).split("-"))
    assert 0.0 <= lo <= hi <= 100.0
