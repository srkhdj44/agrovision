import json
import os
import random
import uuid

from flask import Flask, render_template, request
from PIL import Image, ImageOps

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
# Crop names for the "Supported crops" line, available in every template
app.jinja_env.globals["supported_crops"] = sorted({d["crop"] for d in DISEASES.values()})

def predict_placeholder(image_path):
    names = random.sample(list(DISEASES.keys()), 3)
    first = round(random.uniform(0.30, 0.99), 2)
    rest = 1 - first
    return [(names[0], first), (names[1], round(rest * 0.7, 2)), (names[2], round(rest * 0.2, 2))], None
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
    image = Image.open(file.stream)
    image = ImageOps.exif_transpose(image).convert("RGB")
    saved_name = f"{uuid.uuid4().hex}.jpg"
    saved_path = os.path.join(UPLOAD_FOLDER, saved_name)
    image.save(saved_path, quality=90)

    # Predict and look up the disease information
    if USE_REAL_MODEL:
        results, overlay = predict(saved_path)
    else:
        results, overlay = predict_placeholder(saved_path)
    class_name, confidence = results[0]
    info = DISEASES.get(class_name)

    # Other possibilities: the 2nd and 3rd guesses, if they are at least 1%
    others = []
    for name, score in results[1:]:
        if score >=0.01:
            entry = DISEASES.get(name)
            others.append({
                "name": entry["display_name"] if entry else name,
                "pct": round(score*100),
            })

    # Save the Grad-CAM heatmap next to the upload
    heatmap_file = None
    if overlay is not None:
        heatmap_file = saved_name.replace(".jpg", "_heatmap.jpg")
        overlay.save(os.path.join(UPLOAD_FOLDER, heatmap_file), quality=90)

    return render_template(
        "result.html",
        image_file=saved_name,
        class_name=class_name,
        confidence=confidence,
        info=info,
        others=others,
        low_confidence=confidence < CONFIDENCE_THRESHOLD,
        heatmap_file=heatmap_file,
    )

@app.errorhandler(413)
def file_too_large(error):
    return render_template("upload.html", error="This image is larger than 5MB. Please choose a smaller one"), 413

if __name__ == "__main__":
    app.run(debug=True)