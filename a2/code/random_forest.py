
import numpy as np

from random_tree import RandomTree
from scipy import stats

class RandomForest:
        
    def __init__(self, max_depth, num_trees):
        self.max_depth = max_depth
        self.num_trees = num_trees
        self.stats = []

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.stats = []
        for i in range(self.num_trees):
            self.stats = np.append(self.stats, RandomTree(self.max_depth))
            self.stats[i].fit(X, y)

    def predict(self, X):
        j = X.shape[0]
        y_hat = np.ones((j, self.num_trees), np.uint8)
        for i in range(self.num_trees):
            y_hat[:, i] = self.stats[i].predict(X)

        return stats.mode(y_hat, 1)[0].flatten()


