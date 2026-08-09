"""Keeps the default test run offline.

The plan requires that tests need neither network access nor the real corpora.
The tokenizer tests genuinely need a base model to add tokens to, so they are
marked `network` and skipped unless explicitly requested:

    TYPESHI_NETWORK_TESTS=1 uv run pytest
"""

import os

import pytest

RUN_NETWORK = os.environ.get("TYPESHI_NETWORK_TESTS") == "1"


def pytest_collection_modifyitems(config, items):
    if RUN_NETWORK:
        return
    skip = pytest.mark.skip(reason="needs network; set TYPESHI_NETWORK_TESTS=1 to run")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip)
