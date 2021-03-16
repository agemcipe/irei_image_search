import cv2
from sklearn.cluster import KMeans
import pickle
import pandas as pd
import numpy as np
import multiprocessing as mp


class SIFTDescriptor:
    # Define KMeans settings or load pretrained model.
    def __init__(self, n_clusters=20, model_path=None):
        if model_path is None:
            self.model = KMeans(n_clusters=n_clusters)
            self.n_clusters = n_clusters
        else:
            self.model = pickle.load(open(model_path, "rb"))
            kmeans = KMeans()
            self.n_clusters = kmeans.get_params()['n_clusters']

    # Return the SIFT description for each keypoint found.
    # Return shape: (k, 128) with k = number of keypoints.
    def sift_descriptor(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        sift = cv2.SIFT_create()
        _, description = sift.detectAndCompute(gray, None)

        return description

    # Construct dataframe where each row represents 'image_ID' + feature_vector of single keypoint descriptor.
    # Return shape: (k, 1+128) with k = number of keypoints.
    # Return None if sift_descriptor returns None.
    def get_sift_descriptors_df(self, image_path):
        image_ID = image_path[image_path.rfind("/") + 1:]
        image = cv2.imread(image_path)

        descriptors = self.sift_descriptor(image)
        if descriptors is None:
            print('%s has no sift descriptor.' % image_ID)
            return None
        num_descriptors = descriptors.shape[0]

        id_vector = np.repeat([image_ID], num_descriptors, axis=0)
        id_vector.resize((num_descriptors, 1))
        np_dataframe = np.concatenate((id_vector, descriptors), axis=1)
        df_descriptors = pd.DataFrame(np_dataframe,
                                          columns=['image_ID'] + list(range(1, 129)))
        return df_descriptors

    # Make histogram of BOW
    def make_histogram(self, bow):
        hist, _ = np.histogram(bow, bins=self.n_clusters)
        return hist

    # Get descriptor (histogram) for query image
    def describe(self, image):
        sift_description = self.sift_descriptor(image)
        df_sift = pd.DataFrame(sift_description)
        bag_of_words = self.model.predict(df_sift)

        hist = self.make_histogram(bag_of_words)

        return hist

    # Describe full data set when indexing.
    # Return data frame where each row = image_id + histogram vector
    # If model_path is given, save trained KMeans model as pickle.
    def describe_full_data_set(self, image_paths, model_path=None, verbose=False):
        # Create train set of descriptors to fit KMeans on
        if verbose:
            print('Creating training set for KMeans.')
        pool = mp.Pool(mp.cpu_count())
        dfs = pool.map(self.get_sift_descriptors_df, image_paths)

        # Filter None values
        temp = []
        for df in dfs:
            if df is not None:
                temp.append(df)
        dfs = temp

        # Join dfs
        df_descriptors = pd.concat(dfs)

        df_descriptors = df_descriptors.reset_index()
        if verbose:
            print('Training set ready.')
            print('Training KMeans.')

        # Train KMeans and predict labels
        label = self.model.fit_predict(df_descriptors.drop(['image_ID'], axis=1))
        if verbose:
            print('KMeans trained.')

        # Free memory but keep image ids
        image_ids = df_descriptors['image_ID']
        df_descriptors = None

        df_bow = pd.concat([image_ids, pd.Series(label)], axis=1)

        df_hist_full = None
        for image_ID in image_ids.unique():
            hist = self.make_histogram(df_bow[df_bow['image_ID'] == image_ID][0])
            if df_hist_full is None:
                df_hist_full = pd.DataFrame([hist])
                df_hist_full.insert(loc=0, column='image_ID', value=[image_ID])
            else:
                df_hist_full1 = pd.DataFrame([hist])
                df_hist_full1.insert(loc=0, column='image_ID', value=[image_ID])
                df_hist_full = df_hist_full.append(df_hist_full1)

        if model_path is not None:
            pickle.dump(self.model, open(model_path, "wb"))
            if verbose:
                print('Model saved to %s.' % model_path)

        if verbose:
            print('Index created.')
        return df_hist_full
