# Toy Image-Search-Engine
Content-Based Image Retrieval System for Satellite Images Implemented Using Python, Flask And OpenCV.
* Given a query image, this app returns other images from a directory that are similar.
* "Similarity" is either based on the description of an image with a color histogram or with [SIFT](https://en.wikipedia.org/wiki/Scale-invariant_feature_transform) features. The distance can be calculated using the chi-squared or canberra distance.

## Usage Guide
1. To use a different image dataset (optional)
    * Populate image DB in `app/static/images`
    * Then in Terminal: 
      ```bash
      >> python3 -m venv venv
      >> source venv/bin/activate
      >> pip install -r requirements.txt
      >> cd app
      >> python index.py --dataset static/images --descriptor color --index index_color.csv
      ```

2. Run locally using Docker
    * Install [Docker](https://docs.docker.com/install/#supported-platforms)
    * Then in Terminal:
      ```bash
      >> docker build --tag=imagesearch .
      >> docker run -p 80:8000 imagesearch
      ```
* You should be able to access app at `localhost:80` in browser

### Credits
Project is based on the work of Kene Udeh and the original work can be found [here])(https://github.com/fatemaquaid987/Content_Based_Image_Retrieval)