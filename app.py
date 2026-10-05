"""Flask web application that predicts hand-sign digits from uploaded images."""
from flask import Flask, render_template, request
from PIL import UnidentifiedImageError

from model import preprocess_img, predict_result

app = Flask(__name__)

ERROR_MESSAGE = "File cannot be processed."


@app.route("/")
def main():
    """Render the home page containing the image upload form."""
    return render_template("index.html")


@app.route("/prediction", methods=["POST"])
def predict_image_file():
    """Preprocess the uploaded image and render the predicted digit.

    Any missing, empty, or unreadable upload renders the result page
    with an error message instead of a prediction.
    """
    uploaded = request.files.get("file")
    if uploaded is None or uploaded.filename == "":
        return render_template("result.html", err=ERROR_MESSAGE)

    try:
        img = preprocess_img(uploaded.stream)
        pred = predict_result(img)
    except (UnidentifiedImageError, OSError, ValueError):
        return render_template("result.html", err=ERROR_MESSAGE)

    return render_template("result.html", predictions=str(pred))


if __name__ == "__main__":
    app.run(port=9000, debug=True)
