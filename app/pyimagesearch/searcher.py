# import the necessary packages
import numpy as np
import csv
import scipy.spatial.distance

IMPLEMENTED_METRICS = ["chi2", "canberra"]


class Searcher:
    def __init__(self, indexPath):
        # store our index path
        self.indexPath = indexPath

    def search(self, queryFeatures, limit=101, metric="chi2"):

        if metric not in IMPLEMENTED_METRICS:
            raise ValueError(f"Unknown metric '{metric}'")

        # initialize dictionary of results
        results = {}

        # open the index file
        with open(self.indexPath) as f:
            # initialize the CSV reader
            reader = csv.reader(f)

            # loop over the rows in the index
            for row in reader:
                # parse out the image ID and features, then compute the
                # chi-squared dist. b/w the features in our index
                # and our query features
                features = [float(x) for x in row[1:]]
                if metric == "chi2":
                    d = self.chi2_distance(features, queryFeatures)
                elif metric == "canberra":
                    d = scipy.spatial.distance.canberra(features, queryFeatures)
                else:
                    raise NotImplementedError

                # Update dictionsary
                # key is image id, value is similarity
                results[row[0]] = d

            # close reader
            f.close()

        # sort in order of relevance
        results = sorted([(v, k) for (k, v) in results.items()])

        # return our (limited) results
        return results[:limit]

    def chi2_distance(self, histA, histB, eps=1e-10):
        # compute chi-squared distance
        d = 0.5 * np.sum(
            [((a - b) ** 2) / (a + b + eps) for (a, b) in zip(histA, histB)]
        )

        # return the chi-squared distance
        return d
