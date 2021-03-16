# import the necessary packages
import pandas as pd
from searcher import Searcher
import seaborn as sn
import matplotlib.pyplot as plt
import random


class Evaluator:
    def __init__(self, indexPath):
        # store index path
        self.indexPath = indexPath
        self.lulc_classes = ['industrial_area', 'beach', 'lake', 'circular_farmland', 'sparse_residential',
                   'cloud', 'mountain', 'dense_residential', 'desert', 'forest']

        # Create 'random' test set
        random.seed(42)
        images_numbers = random.sample(range(1, 101), 10)  # For test purposes hardcoded.

        image_names = []
        for lulc_class, number in [(a, b) for a in self.lulc_classes for b in images_numbers]:
            if number < 10:
                num_string = '00' + str(number)
            elif number < 100:
                num_string = '0' + str(number)
            else:
                num_string = str(number)
            image_names.append(lulc_class + '_' + num_string + '.jpg')
        self.image_names = image_names

    def evaluate(self, queryFeatures, limit=10, image_in_index=False):
        # initialize Searcher
        searcher = Searcher(self.indexPath)

        # retrieve results
        # If query image is indexed, it will be found first with distance = 0.
        if image_in_index:
            results = searcher.search(queryFeatures=queryFeatures, limit=limit + 1)[1:]
        else:
            results = searcher.search(queryFeatures=queryFeatures, limit=limit)

        return strip_classes(results)

    def evaluate_all(self, image_names, limit=10, image_in_index=False):
        # initialize Searcher
        searcher = Searcher(self.indexPath)

        # open the index file
        with open(self.indexPath) as f:
            retrieved_classes = []
            actual_classes = []

            df = pd.read_csv(self.indexPath, header=None)
            df = df[df[0].isin(image_names)]
            df[0] = df[0].transform(lambda x: x.rsplit('_', 1)[0])

            for _, row in df.iterrows():
                retrieved_classes += self.evaluate(row[1:], limit, image_in_index=image_in_index)
                actual_classes += [row[0]] * limit

        data = {'class_actual': actual_classes,
                'class_retrieved': retrieved_classes
                }

        df = pd.DataFrame(data, columns=['class_actual', 'class_retrieved'])

        return df


def strip_classes(results):
    # get retrieved classes from results
    retrieved_classes = []
    for _, k in results:
        retrieved_classes.append(k.rsplit('_', 1)[0])

    return retrieved_classes


# The main method is for testing purposes
# ToDo: Move to evaluation Jupyter notebook
if __name__ == '__main__':
    descriptor = 'sift'
    df = None
    if descriptor == 'color':
        evaluator = Evaluator('/home/till/PycharmProjects/irei_image_search2/app/index_color.csv')

        df = evaluator.evaluate_all(evaluator.image_names, limit=10, image_in_index=True)

        df.to_csv('evaluation_10_colordescriptor.csv')

    elif descriptor == 'sift':
        evaluator = Evaluator('/home/till/PycharmProjects/irei_image_search2/app/index_sift.csv')
        df = evaluator.evaluate_all(evaluator.image_names, limit=10, image_in_index=True)

        df.to_csv('evaluation_10_siftdescriptor.csv')

    cm = pd.crosstab(df['class_actual'], df['class_retrieved'], rownames=['Actual'], colnames=['Retrieved'])
    sn.heatmap(cm, annot=True)
    plt.show()
