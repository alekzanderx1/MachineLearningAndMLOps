"""

Simple stand-alone script to create a txt file representing a fake datasets 
to train a classification model later on.

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import OrderedDict
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier


def plot_scatter(x: list, y: list):
    plt.title('X Vs. Y')
    plt.plot(x, y, '.', label='dataset')
    plt.savefig('classification.png', bbox_inches='tight')

    return


def main():
    x, y = make_classification(
        n_samples=1000,  # number of rows
        n_features=6, # number of features
        n_informative=3, # The number of informative features
        n_redundant = 2, # The number of redundant features
        n_repeated = 1, # The number of duplicated features
        n_classes = 2, # The number of classes 
        n_clusters_per_class=1,#The number of clusters per class
        random_state = 42 # random seed 
    )
    
    # dump data to a local txt file
    with open('classification_dataset.txt', 'w') as f:
        for _x, _y in zip(x, y):
            f.write('{}\t{}\n'.format("\t".join([str(val) for val in _x]), _y))
    # plot it for visual inspection
    plot_scatter(x, y)
    return


if __name__ == "__main__":
    main()