"""Minimal smoke tests so CI pytest step passes."""

import pytest


def test_import_core():
    """Core module imports without error."""
    import core.pipeline  # noqa: F401
    assert True


def test_import_modules():
    """Modules package imports without error."""
    import modules  # noqa: F401
    assert True
