"""Tests for the Flask routes in app.py."""
import io

import app as app_module
from tests.conftest import make_image_bytes


def test_home_page_loads(client):
    """The home page returns 200 and shows the upload form."""
    response = client.get("/")
    assert response.status_code == 200
    assert b"Hand Sign Digit Language Detection" in response.data
    assert b'name="file"' in response.data


def test_valid_upload_returns_prediction(client, monkeypatch):
    """A valid image upload renders the predicted digit (model mocked)."""
    monkeypatch.setattr(app_module, "predict_result", lambda img: 7)
    data = {"file": (make_image_bytes(), "digit.png")}
    response = client.post("/prediction", data=data, content_type="multipart/form-data")
    assert response.status_code == 200
    assert b">7</h2>" in response.data
    assert app_module.ERROR_MESSAGE.encode() not in response.data


def test_missing_file_shows_error(client):
    """A POST with no file part shows the error message instead of crashing."""
    response = client.post("/prediction", data={}, content_type="multipart/form-data")
    assert response.status_code == 200
    assert app_module.ERROR_MESSAGE.encode() in response.data


def test_empty_filename_shows_error(client):
    """Submitting the form without choosing a file is handled as missing input."""
    data = {"file": (io.BytesIO(b""), "")}
    response = client.post("/prediction", data=data, content_type="multipart/form-data")
    assert app_module.ERROR_MESSAGE.encode() in response.data


def test_non_image_file_shows_error(client, monkeypatch):
    """A text file is rejected before the model is ever called."""
    called = []
    monkeypatch.setattr(app_module, "predict_result", called.append)
    data = {"file": (io.BytesIO(b"this is not an image"), "notes.txt")}
    response = client.post("/prediction", data=data, content_type="multipart/form-data")
    assert app_module.ERROR_MESSAGE.encode() in response.data
    assert not called


def test_get_on_prediction_route_not_allowed(client):
    """The prediction route only accepts POST requests."""
    response = client.get("/prediction")
    assert response.status_code == 405
