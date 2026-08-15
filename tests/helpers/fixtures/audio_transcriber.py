"""Fixtures for the reused Whisper transcription worker."""

import sys

import pytest


@pytest.fixture(autouse=True, scope="module")
def release_audio_transcriber_after_module():
    """Free the transcription worker's device memory once a test module is done.

    The worker holds a loaded Whisper model, so leaving it up would keep
    accelerator memory occupied while a later module starts its own server.
    Reached through ``sys.modules`` because ``tests/conftest.py`` deliberately
    keeps ``tests.helpers.media`` lazy: a module that never transcribed should
    not pay to import it.
    """
    yield
    media = sys.modules.get("tests.helpers.media")
    if media is not None:
        media.release_audio_transcriber()
