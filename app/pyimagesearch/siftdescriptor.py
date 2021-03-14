# import the necessary packages
import cv2
import config
import glob

import matplotlib.pyplot as plt


class SIFTDescriptor:
    def describe(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        sift = cv2.SIFT_create()
        keypoint, descriptor = sift.detectAndCompute(gray, None)

        img = cv2.drawKeypoints(gray, keypoint, image, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # imgplot = plt.imshow(img)
        # plt.show()

        return descriptor


if __name__ == '__main__':
    image = cv2.imread('/home/till/PycharmProjects/irei_image_search/app/input/church/church_001.jpg')

    siftDescriptor = SIFTDescriptor()
    descr = siftDescriptor.describe(image)
    print(descr)

    lulc_classes = config.LULC_CLASSES_10
    dataset = '/home/till/PycharmProjects/irei_image_search/app/input'
    # open the output index file for writing
    output = open('sift_descriptors_lulc10.csv', "w")
    types = ('/*.jpg', '/*.png', '/*.gif')  # the tuple of file types
    files_grabbed = []
    for files in types:
        for lulc in lulc_classes:
            files_grabbed.extend(glob.glob(dataset + '/' + lulc + files))

    # use glob to grab the image paths and loop over them
    for imagePath in files_grabbed:
        # extract the imageID from the image
        # path and load the image itself
        imageID = imagePath[imagePath.rfind("/") + 1:]
        image = cv2.imread(imagePath)

        # describe the image
        features = siftDescriptor.describe(image)

        # write the features to file
        features = [str(f) for f in features]
        output.write("%s,%s\n" % (imageID, ",".join(features)))

        print(imagePath)

    # close the index file
    output.close()
