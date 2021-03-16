# import the necessary packages
from pyimagesearch.colordescriptor import ColorDescriptor
from pyimagesearch.siftdescriptor import SIFTDescriptor
import argparse
import glob
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to the directory that contains the images to be indexed")
ap.add_argument("-i", "--index", required=True,
                help="path to where the index should be stored")
ap.add_argument("-s", "--descriptor", default='color',
                help="which descriptor is to be used: 'color' or 'sift'")
args = vars(ap.parse_args())

if args["descriptor"] == 'color':
    # initialize the color descriptor
    cd = ColorDescriptor((8, 12, 3))

    # open the output index file for writing
    output = open(args["index"], "w")
    types = ('/*.jpg', '/*.png', '/*.gif')  # the tuple of file types
    files_grabbed = []
    for files in types:
        files_grabbed.extend(glob.glob(args["dataset"] + files))

    # use glob to grab the image paths and loop over them
    for imagePath in files_grabbed:
        # extract the imageID from the image
        # path and load the image itself
        imageID = imagePath[imagePath.rfind("/") + 1:]
        image = cv2.imread(imagePath)

        # describe the image
        features = cd.describe(image)

        # write the features to file
        features = [str(f) for f in features]
        output.write("%s,%s\n" % (imageID, ",".join(features)))

    # close the index file
    output.close()

elif args["descriptor"] == 'sift':
    # initialize the color descriptor
    sd = SIFTDescriptor(20)

    # grab files
    types = ('/*.jpg', '/*.png', '/*.gif')  # the tuple of file types
    files_grabbed = []
    for files in types:
        files_grabbed.extend(glob.glob(args["dataset"] + files))

    index = sd.describe_full_data_set(files_grabbed, 'kmeans_lulc_10.pkl', verbose=True)

    index.to_csv(args["index"], header=False, index=False)
