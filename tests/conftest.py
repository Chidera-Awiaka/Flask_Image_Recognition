"""Shared pytest fixtures for the Flask image recognition tests."""
import io

import pytest
from PIL import Image

from app import app as flask_app


@pytest.fixture(name="client")
def fixture_client():
    """Provide a Flask test client with testing mode enabled."""
    flask_app.config["TESTING"] = True
    with flask_app.test_client() as test_client:
        yield test_client


def make_image_bytes(mode="RGB", size=(64, 64), fmt="PNG"):
    """Create an in-memory image file so tests do not depend on disk files."""
    colour = 128 if mode == "L" else (128,) * len(mode)
    buffer = io.BytesIO()
    Image.new(mode, size, colour).save(buffer, format=fmt)
    buffer.seek(0)
    return buffer
