"""Shared pytest configuration.

Patches the default store singleton so tests never write to disk.
"""

from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _no_default_store():
    """Prevent the global Store singleton from creating petal.db during tests."""
    with patch("petal.store._default_store", None):
        with patch("petal.store.get_default_store", return_value=None):
            yield
