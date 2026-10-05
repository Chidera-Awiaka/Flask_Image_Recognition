"""Tests for image preprocessing and prediction in model.py."""
import io

import numpy as np
import pytest
from PIL import UnidentifiedImageError

import model
from tests.conftest import make_image_bytes

SAMPLE_IMAGE = "test_images/3/Sign 3 (30).jpeg"


def test_preprocess_returns_expected_shape_and_range():
    """A real sample image becomes a (1, 224, 224, 3) array scaled to 0-1."""
    result = model.preprocess_img(SAMPLE_IMAGE)
    assert result.shape == (1, 224, 224, 3)
    assert result.min() >= 0.0
    assert result.max() <= 1.0


@pytest.mark.parametrize("mode", ["RGB", "RGBA", "L"])
def test_preprocess_handles_colour_modes(mode):
    """PNGs with transparency and greyscale images are converted to 3 channels."""
    result = model.preprocess_img(make_image_bytes(mode=mode))
    assert result.shape == (1, 224, 224, 3)


def test_preprocess_rejects_non_image_bytes():
    """Non-image data raises an error the Flask route can catch."""
    with pytest.raises(UnidentifiedImageError):
        model.preprocess_img(io.BytesIO(b"not an image"))


def test_predict_result_returns_highest_class(monkeypatch):
    """predict_result returns the index of the highest score (model mocked)."""

    class FakeModel:  # pylint: disable=too-few-public-methods
        """Stand-in for the Keras model with a fixed output."""

        def predict(self, _batch):
            """Return scores where class 4 is the most likely."""
            scores = np.zeros((1, 10))
            scores[0, 4] = 0.9
            return scores

    monkeypatch.setattr(model, "get_model", FakeModel)
    assert model.predict_result(np.zeros((1, 224, 224, 3))) == 4


@pytest.mark.slow
def test_real_model_predicts_sample_digit():
    """End-to-end check: the trained model classifies a sample sign 3 image."""
    prediction = model.predict_result(model.preprocess_img(SAMPLE_IMAGE))
    assert int(prediction) == 3
