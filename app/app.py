import os
import cv2
import numpy as np


from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from werkzeug.utils import secure_filename

from pyimagesearch.colordescriptor import ColorDescriptor
from pyimagesearch.siftdescriptor import SIFTDescriptor
from pyimagesearch.searcher import Searcher, IMPLEMENTED_METRICS
from pyimagesearch import IMPLEMENTED_DESCRIPTORS

VERBOSE = False  # only for debugging purposes


IMAGE_DIR = os.path.join(os.path.dirname(__file__), "static/images/")
ALLOWED_EXTENSIONS = {"jpg", "png", "jpeg"}
DESCRIPTOR_MAP = {"ColorHistogram": "color", "SIFT": "sift"}

############## Helper functions ##############
def _get_index(descriptor: str):
    return os.path.join(os.path.dirname(__file__), f"index_{descriptor}.csv")


def _allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# create flask instance
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = IMAGE_DIR
app.config["MAX_CONTENT_PATH"] = 2 ** 10
app.config["SECRET_KEY"] = "12345"


# main route
@app.route("/", methods=["GET", "POST"])
def index():
    return render_template(
        "index.html",
        preview="static/init-preview.png",
        metrics=IMPLEMENTED_METRICS,
        descriptors=IMPLEMENTED_DESCRIPTORS,
    )


# image database url list route
@app.route("/list", methods=["POST"])
def image_list():

    if request.method == "POST":

        try:

            img_list = sorted(
                [
                    img
                    for img in list(os.listdir(IMAGE_DIR))
                    if img[-4:] in (".png", ".jpg", ".gif")
                ],
                key=lambda x: x.lower()[:6],
            )

            return jsonify(imgList=img_list)

        except Exception as e:
            return jsonify({"sorry": "Sorry, no results! Please try again."}), 500


# search route
@app.route("/search", methods=["POST"])
def search():

    if request.method == "POST":

        RESULTS_ARRAY = []
        descriptor = DESCRIPTOR_MAP[request.form.get("descriptor")]
        # get url
        image_url = request.form.get("img")

        try:
            # load the query image and describe it
            from skimage import io
            import cv2

            query = cv2.imread(
                os.path.join(os.path.dirname(__file__), "static/images/" + image_url)
            )

            index_used = _get_index(descriptor)

            if descriptor == "color":
                # initialize the image descriptor
                descr = ColorDescriptor((8, 12, 3))
            elif descriptor == "sift":
                descr = SIFTDescriptor(20, model_path="kmeans_lulc_10.pkl")

            features = descr.describe(query)

            # perform the search
            searcher = Searcher(index_used)

            metric = request.form.get("metric")

            if VERBOSE:
                print("descriptor:", descriptor)
                print("index:", index_used)
                print("metric:", metric)
            results = searcher.search(features, metric=metric)

            # loop over the results, displaying the score and image name
            for (score, resultID) in results:
                RESULTS_ARRAY.append({"image": str(resultID), "score": str(score)})
            # return success
            return jsonify(results=(RESULTS_ARRAY[:101]), preview="images/" + image_url)

        except Exception as e:
            print(str(e))
            # return error
            return jsonify({"sorry": "Sorry, no results! Please try again."}), 500


@app.route("/uploader", methods=["POST"])
def upload_file():
    if request.method == "POST":
        # check if the post request has the file part
        if "file" not in request.files:
            flash("No file part")
            return redirect(url_for("index"))
        file = request.files["file"]
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == "":
            flash("No selected file")
            return redirect("/")
            # return redirect(request.url)
        if file and _allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(
                os.path.join(app.config["UPLOAD_FOLDER"], "__upload_" + filename)
            )  # add marker for user file
            flash(f"Upload of {filename} successful!")
            return redirect(url_for("index"))


# run!
if __name__ == "__main__":
    app.run("0.0.0.0", debug=True)
