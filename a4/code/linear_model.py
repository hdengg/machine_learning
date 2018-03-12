import numpy as np
from numpy.linalg import solve
import findMin
from scipy.optimize import approx_fprime
import utils

class logReg:
    # Logistic Regression
    def __init__(self, verbose=0, maxEvals=100):
        self.verbose = verbose
        self.maxEvals = maxEvals
        self.bias = True

    def funObj(self, w, X, y):
        yXw = y * X.dot(w)

        # Calculate the function value
        f = np.sum(np.log(1. + np.exp(-yXw)))

        # Calculate the gradient value
        res = - y / (1. + np.exp(yXw))
        g = X.T.dot(res)

        return f, g

    def fit(self,X, y):
        n, d = X.shape

        # Initial guess
        self.w = np.zeros(d)
        utils.check_gradient(self, X, y)
        (self.w, f) = findMin.findMin(self.funObj, self.w,
                                      self.maxEvals, X, y, verbose=self.verbose)
    def predict(self, X):
        return np.sign(X@self.w)


class logRegL0(logReg):
    # L0 Regularized Logistic Regression
    def __init__(self, L0_lambda=1.0, verbose=2, maxEvals=400):
        self.verbose = verbose
        self.L0_lambda = L0_lambda
        self.maxEvals = maxEvals

    def fit(self, X, y):
        n, d = X.shape
        minimize = lambda ind: findMin.findMin(self.funObj,
                                                  np.zeros(len(ind)),
                                                  self.maxEvals,
                                                  X[:, ind], y, verbose=0)
        selected = set()
        selected.add(0)
        minLoss = np.inf
        oldLoss = 0
        bestFeature = -1

        while minLoss != oldLoss:
            oldLoss = minLoss
            print("Epoch %d " % len(selected))
            print("Selected feature: %d" % (bestFeature))
            print("Min Loss: %.3f\n" % minLoss)

            self.w = np.zeros(d)
            for i in range(d):
                if i in selected:
                    continue

                selected_new = selected | {i}
                # then compute the loss and update the minLoss/bestFeature

                self.w[list(selected_new)], loss = minimize(list(selected_new))
                if loss < minLoss:
                    minLoss = loss
                    bestFeature = i

            selected.add(bestFeature)

        self.w = np.zeros(d)
        self.w[list(selected)], _ = minimize(list(selected))

class logRegL2:
    # Logistic Regression L2
    def __init__(self, verbose=1, lammy=1.0, maxEvals=100):
        self.verbose = verbose
        self.lammy = lammy
        self.maxEvals = maxEvals

    def funObj(self, w, X, y):
        yXw = y * X.dot(w)
        lammy = self.lammy

        # Calculate the function value
        f = np.sum(np.log(1. + np.exp(-yXw))) + (w.T.dot(w) * lammy / 2)

        # Calculate the gradient value
        res = - y / (1. + np.exp(yXw))
        g = X.T.dot(res) + (lammy * w)

        return f, g

    def fit(self, X, y):
        n, d = X.shape

        # Initial guess
        self.w = np.zeros(d)
        utils.check_gradient(self, X, y)
        (self.w, f) = findMin.findMin(self.funObj, self.w,
                                      self.maxEvals, X, y, verbose=self.verbose)

    def predict(self, X):
        w = self.w
        yhat = np.dot(X, w)

        return np.sign(yhat)


class logRegL1:
    # Logistic Regression
    def __init__(self, lammy, verbose=1, maxEvals=100):
        self.lammy = lammy
        self.verbose = verbose
        self.maxEvals = maxEvals

    def funObj(self, w, X, y):
        yXw = y * X.dot(w)

        # Calculate the function value
        f = np.sum(np.log(1. + np.exp(-yXw))) + (self.lammy) * np.sum(w)

        # Calculate the gradient value
        res = - y / (1. + np.exp(yXw))
        g = X.T.dot(res) + self.lammy
        return f, g

    def fit(self, X, y):
        n, d = X.shape

        # Initial guess
        self.w = np.zeros(d)
        utils.check_gradient(self, X, y)
        (self.w, f) = findMin.findMinL1(self.funObj, self.w, self.lammy,
                                      self.maxEvals, X, y, verbose=self.verbose)

    def predict(self, X):
        w = self.w
        yhat = np.dot(X, w)
        return np.sign(yhat)

class leastSquaresClassifier:
    def fit(self, X, y):
        n, d = X.shape
        self.n_classes = np.unique(y).size

        # Initial guess
        self.W = np.zeros((self.n_classes,d))

        for i in range(self.n_classes):
            ytmp = y.copy().astype(float)
            ytmp[y==i] = 1
            ytmp[y!=i] = -1

            # solve the normal equations
            # with a bit of regularization for numerical reasons
            self.W[i] = np.linalg.solve(X.T@X+0.0001*np.eye(d), X.T@ytmp)

    def predict(self, X):
        return np.argmax(X@self.W.T, axis=1)

class logLinearClassifier:
    def __init__(self, maxEvals, verbose):
        self.verbose = verbose
        self.maxEvals = maxEvals

    def funObj(self, w, X, y):
        yXw = y * X.dot(w)

        # Calculate the function value
        f = np.sum(np.log(1. + np.exp(-yXw)))

        # Calculate the gradient value
        res = - y / (1. + np.exp(yXw))

        g = X.T.dot(res)

        return f, g

    def fit(self, X, y):
        n, d = X.shape
        self.n_classes = np.unique(y).size

        # Initial guess
        self.W = np.zeros((d, self.n_classes))

        for i in range(self.n_classes):
            ytmp = y.copy().astype(float)
            ytmp[y==i] = 1
            ytmp[y!=i] = -1

            # solve normal eqn
            (self.W[:, i], f) = findMin.findMin(self.funObj, self.W[:, i],
                                            self.maxEvals, X, ytmp, verbose=self.verbose)

    def predict(self, X):
        yhat = np.dot(X, self.W)

        return np.argmax(yhat, axis=1)

class softmaxClassifier:
    def __init__(self, maxEvals, verbose):
        self.verbose = verbose
        self.maxEvals = maxEvals

    def funObj(self, w, X, y):
        n, d = X.shape
        W = np.reshape(w, (d, self.n_classes))

        y1 = np.zeros(W.shape)
        y2 = np.zeros(W.shape)
        Xw = X.dot(W)
        wyx = np.zeros((n,))
        prob = np.ones((n,))

        for i in range(n):
            wyx[i] = Xw[i, y[i]]

        for i in np.unique(y):
            y1[:, i] = -np.sum(X[y == i], axis=0)
            # predicted probability
            divisor = np.sum(np.exp(Xw), axis=1)
            numerator = np.exp(Xw[:, i])
            prob[:] = numerator[:] / divisor[:]
            y2[:, i] = prob.dot(X)

        # compute softmax loss given formula in assignment
        f = -np.sum(wyx) + np.sum(np.log(np.sum(np.exp(Xw), axis=1)))

        g = y1 + y2

        return f, g.ravel()

    def fit(self, X, y):
        n, d = X.shape
        self.n_classes = np.unique(y).size

        # Initial guess
        self.w = np.zeros(d*self.n_classes)
        utils.check_gradient(self, X, y)

        (self.w, f) = findMin.findMin(self.funObj, self.w.flatten(),
                                      self.maxEvals, X, y, verbose=self.verbose)

        self.w = np.reshape(self.w, (d, self.n_classes))

    def predict(self, X):
        yhat = np.dot(X, self.w)

        return np.argmax(yhat, axis=1)

