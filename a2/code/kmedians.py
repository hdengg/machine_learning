import numpy as np
import utils


class Kmedians:

    def __init__(self, k):
        self.k = k

    def fit(self, X):
        N, D = X.shape
        y = np.ones(N)

        median = np.zeros((self.k, D))
        for kk in range(self.k):
            i = np.random.randint(N)
            median[kk] = X[i]

        while True:
            y_old = y

            # Compute the L1 distance to center
            l1_dist = np.zeros((N, self.k))
            for i in range(N):
                for kk in range(self.k):
                    # the L1 norm
                    l1_dist[i][kk] = np.sum(np.abs(X[i] - median[kk]))
            y = np.argmin(l1_dist, 1)

            # Update the medians
            for kk in range(self.k):
                if X[y == self.k].size != 0:
                    median[kk] = np.median(X[y == kk], 0)

            # Stop if there are no changes
            changes = np.sum(y != y_old)
            # print('Running K-medians, changes in cluster assignment = {}'.format(changes))

            # Stop if no point changed cluster
            if changes == 0:
                break

        self.median = median

    def predict(self, X):
        median = self.median
        N = X.shape[0]
        self.k = median.shape[0]
        l1_dist = np.zeros((N, self.k))

        for i in range(N):
            for kk in range(self.k):
                l1_dist[i][kk] = np.sum(np.abs(X[i] - median[kk]))

        return np.argmin(l1_dist, 1)

    def error(self, X):
        median = self.median
        y = self.predict(X)

        error = 0
        for kk in range(median.shape[0]):
            X_k = X[y == kk]
            for i in range(len(X_k)):
                error += np.sum(np.abs(X_k[i] - median[kk]))

        return error




