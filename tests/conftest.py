"""Keeps the default test run offline.

The plan requires that tests need neither network access nor the real corpora.
The tokenizer tests genuinely need a base model to add tokens to, so they are
marked `network` and skipped unless explicitly requested:

    TYPESHI_NETWORK_TESTS=1 uv run pytest
"""

import os

import pytest

RUN_NETWORK = os.environ.get("TYPESHI_NETWORK_TESTS") == "1"
RUN_SLOW = os.environ.get("TYPESHI_SLOW_TESTS") == "1"


def pytest_collection_modifyitems(config, items):
    skips = {}
    if not RUN_NETWORK:
        skips["network"] = pytest.mark.skip(
            reason="needs network; set TYPESHI_NETWORK_TESTS=1 to run")
    if not RUN_SLOW:
        skips["slow"] = pytest.mark.skip(
            reason="trains for minutes; set TYPESHI_SLOW_TESTS=1 to run")
    for item in items:
        for name, mark in skips.items():
            if name in item.keywords:
                item.add_marker(mark)
