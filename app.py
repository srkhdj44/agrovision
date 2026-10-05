from flask import Flask, render_template, request
from PIL import Image

app = Flask(__name__)

# Reject any upload bigger than 5MB before it reaches our code
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

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

    # All checks passed. The model will go here in week 4.
    return render_template("upload.html", success=f"Image received: {file.filename}")


@app.errorhandler(413)
def file_too_large(error):
    return render_template("upload.html", error="This image is larger than 5MB. Please choose a smaller one."), 413


if __name__ == "__main__":
    app.run(debug=True)