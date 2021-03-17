import pandas as pd
import argparse

if __name__ == "__main__":
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-a",
                    "--index1",
                    required=True,
                    help="path to the first index")
    ap.add_argument("-b",
                    "--index2",
                    required=True,
                    help="path to the second index")
    ap.add_argument("-o",
                    "--out",
                    required=True,
                    help="where the concatenated index should be stored")
    args = vars(ap.parse_args())

    df_color = pd.read_csv(args["index1"], header=None,index_col=0)
    df_sift = pd.read_csv(args["index2"], header=None,index_col=0)

    mean_color = df_color.mean(axis=1).mean()
    mean_sift = df_sift.mean(axis=1).mean()

    df_color /= mean_color
    df_sift /= mean_sift

    df_color_sift = pd.concat([df_color, df_sift], axis=1,join='inner')

    df_color_sift.to_csv(args["out"],header=None)