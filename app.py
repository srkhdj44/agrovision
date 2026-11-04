import json
import os
import random
import uuid

from flask import Flask, render_template, request
from PIL import Image

app = Flask(__name__)

# Reject any upload bigger than 5MB before it reaches our code
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
# Below this confidence, the app says it is not sure instead of showing a disease.
# 0.60 is a starting value. Tune in with Afiqah once the real model is in.
CONFIDENCE_THRESHOLD = 0.60

# True = use Afiqah's model. False= use the random Placeholder (no tensorflow needed)
USE_REAL_MODEL = True

if USE_REAL_MODEL:
    from predictor import predict

# Folder paths, based on where app.py is
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the disease information once, when the app starts
with open (os.path.join(BASE_DIR, "data", "diseases.json"), encoding="utf-8") as f:
    DISEASES = json.load(f)

def predict_placeholder(image_path):
    """Temporary stand-in for the real model.
    Picks a random class that has information with a confidence.
    Replace this with the real model in week 6."""
    class_name = random.choice(list(DISEASES.keys()))
    confidence = round(random.uniform(0.30, 0.99), 2)
    return class_name, confidence

def allowed_file(filename):
    """Return True if the filename ends with .jpg, .jpeg or .png."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/how-it-works")
def how_it_works():
    return render_template("how_it_works.html")

@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "GET":
        return render_template("upload.html")

    # ----- POST: the user submitted the form -----
    file = request.files.get("leaf_image")

    # Check 1: was a file sent at all?
    if file is None or file.filename == "":
        return render_template("upload.html", error="Please choose an image first.")

    # Check 2: is the extension allowed?
    if not allowed_file(file.filename):
        return render_template("upload.html", error="Please choose a JPG, JPEG or PNG image.")

    # Check 3: is it really an image (not a renamed file or a broken one)?
    try:
        image = Image.open(file.stream)
        image.verify()
    except Exception:
        return render_template("upload.html", error="This file is not a valid image. Please try another one.")

    # All checks passed. save the image with a unique name
    file.stream.seek(0)
    extension = file.filename.rsplit(".",1)[1].lower()
    saved_name = f"{uuid.uuid4().hex}.{extension}"
    saved_path = os.path.join(UPLOAD_FOLDER, saved_name) 
    file.save(saved_path)

    # Predict (placeholder for now) and look up the disease information
    if USE_REAL_MODEL:
        class_name, confidence = predict(saved_path)
    else:
        class_name, confidence = predict_placeholder(saved_path)
    info = DISEASES.get(class_name)

    return render_template(
        "result.html",
        image_file=saved_name,
        class_name=class_name,
        confidence=confidence,
        info=info,
        low_confidence=confidence < CONFIDENCE_THRESHOLD,
    )

@app.errorhandler(413)
def file_too_large(error):
    return render_template("upload.html", error="This image is larger than 5MB. Please choose a smaller one"), 413

if __name__ == "__main__":
    app.run(debug=True)