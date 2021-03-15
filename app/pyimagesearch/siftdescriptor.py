import cv2
from sklearn.cluster import KMeans
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
import multiprocessing as mp


class SIFTDescriptor:
    def __init__(self, n_clusters=20, model_path=None):
        self.n_clusters = n_clusters
        if model_path is None:
            self.model = KMeans(n_clusters=self.n_clusters)
        else:
            self.model = pickle.load(open(model_path, "rb"))

    def sift_descriptor(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        sift = cv2.SIFT_create()
        _, description = sift.detectAndCompute(gray, None)

        return description

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

    def make_histogram(self, bow):
        hist, _ = np.histogram(bow, bins=self.n_clusters)
        return hist

    def describe(self, image):
        sift_description = self.sift_descriptor(image)
        df_sift = pd.DataFrame(sift_description)
        bag_of_words = self.model.predict(df_sift)

        hist = self.make_histogram(bag_of_words)

        return hist

    def describe_full_data_set(self, image_paths, model_path=None):
        # create train set of descriptors to fit KMeans on
        df_descriptors = None
        pool = mp.Pool(mp.cpu_count())
        # dfs = [pool.apply(self.get_sift_descriptors_df, args=image_path) for image_path in image_paths]
        dfs = pool.map(self.get_sift_descriptors_df, image_paths)

        # Filter for None values
        temp = []
        for df in dfs:
            if df is not None:
                temp.append(df)
        dfs = temp

        # Join dfs
        df_descriptors = pd.concat(dfs)

        # for image_path in tqdm(image_paths):
        #     image_ID = image_path[image_path.rfind("/") + 1:]
        #     image = cv2.imread(image_path)
        #
        #     descriptors = self.sift_descriptor(image)
        #     if descriptors is None:
        #         print('%s has no sift descriptor.' % image_ID)
        #         continue
        #     num_descriptors = descriptors.shape[0]
        #
        #     id_vector = np.repeat([image_ID], num_descriptors, axis=0)
        #     id_vector.resize((num_descriptors, 1))
        #     np_dataframe = np.concatenate((id_vector, descriptors), axis=1)
        #     if df_descriptors is None:
        #         df_descriptors = pd.DataFrame(np_dataframe,
        #                                       columns=['image_ID']+list(range(1, 129)))
        #     else:
        #         df_descriptors_new = pd.DataFrame(np_dataframe,
        #                                           columns=['image_ID'] + list(range(1, 129)))
        #         df_descriptors = df_descriptors.append(df_descriptors_new)
        df_descriptors = df_descriptors.reset_index()
        print('Training set ready.')

        # Train kmeans and predict labels ToDo: Delete duplicate model definition
        self.model = KMeans(n_clusters=self.n_clusters)
        label = self.model.fit_predict(df_descriptors.drop(['image_ID'], axis=1))
        print('KMeans trained.')

        # Free memory
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

        print('Index created.')
        return df_hist_full


# Some development code. ToDo: Delete for production
if __name__ == '__main__':
    image_paths = ['/home/till/PycharmProjects/irei_image_search/app/input/airplane/airplane_001.jpg',
                   '/home/till/PycharmProjects/irei_image_search/app/input/airplane/airplane_002.jpg',
                   '/home/till/PycharmProjects/irei_image_search/app/input/airplane/airplane_003.jpg',
                   '/home/till/PycharmProjects/irei_image_search/app/input/airplane/airplane_004.jpg',
                   '/home/till/PycharmProjects/irei_image_search/app/input/airplane/airplane_005.jpg']

    sift_descriptor = SIFTDescriptor(20)

    df = sift_descriptor.describe_full_data_set(image_paths)
    print(df.shape[0])

    print(df)
