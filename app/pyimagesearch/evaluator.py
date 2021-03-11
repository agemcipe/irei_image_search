# import the necessary packages
import pandas as pd
import csv
from searcher import Searcher
import seaborn as sn
import matplotlib.pyplot as plt


CLASS_SAMPLE_SIZE = 3


class Evaluator:
    def __init__(self, indexPath):
        # store our index path
        self.indexPath = indexPath

    def evaluate(self, queryFeatures, limit=10):
        # initialize Searcher
        searcher = Searcher(self.indexPath)

        # retrieve results
        results = searcher.search(queryFeatures=queryFeatures, limit=limit)

        return strip_classes(results)

    def evaluate_all(self, limit=10):
        # initialize Searcher
        searcher = Searcher(self.indexPath)

        # open the index file
        with open(self.indexPath) as f:
            # initialize the CSV reader
            reader = csv.reader(f)

            retrieved_classes = []
            actual_classes = []

            df = pd.read_csv(self.indexPath, header=None)
            df[0] = df[0].transform(lambda x: x.rsplit('_', 1)[0])

            # Take stratified sample
            for _, row in df.groupby(0, group_keys=False).apply(lambda x: x.sample(min(len(x), CLASS_SAMPLE_SIZE))).iterrows():
                retrieved_classes += self.evaluate(row[1:], limit+1)[1:]  # Retrieve limit+1 images, because first
                                                                           # image is always the query image
                actual_classes += [row[0]] * limit

        data = {'class_actual': actual_classes,
                'class_retrieved': retrieved_classes
                }

        df = pd.DataFrame(data, columns=['class_actual', 'class_retrieved'])
        confusion_matrix = pd.crosstab(df['class_actual'], df['class_retrieved'], rownames=['Actual'], colnames=['Retrieved'])


        return confusion_matrix


def strip_classes(results):
    # get retrieved classes from results
    retrieved_classes = []
    for _, k in results:
        retrieved_classes.append(k.rsplit('_', 1)[0])

    return retrieved_classes


if __name__ == '__main__':
    evaluator = Evaluator('/home/till/PycharmProjects/irei_image_search/app/my_index.csv')

    cm = evaluator.evaluate_all(limit=10)

    sn.heatmap(cm, annot=True)
    plt.show()
