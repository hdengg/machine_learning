import numpy as np
from numpy.linalg import solve
from findMin import findMin
from scipy.optimize import approx_fprime
import utils

# Ordinary Least Squares
class LeastSquares:
    def fit(self,X,y):
        self.w = solve(X.T@X, X.T@y)

    def predict(self, X):
        return X@self.w

# Least squares where each sample point X has a weight associated with it.
class WeightedLeastSquares(LeastSquares): # inherits the predict() function from LeastSquares
    def fit(self,X,y,z):
        ''' YOUR CODE HERE '''
        Z = np.diag(z)

        # m = X.T * Z * X
        m = np.dot(np.dot(X.T, Z), X)

        # n = X.T * Z * y
        n = np.dot(np.dot(X.T, Z), y)

        self.w = solve(m, n)

    def predict(self, X):
        w = self.w
        yhat = np.dot(X, w)
        return yhat


class LinearModelGradient(LeastSquares):

    def fit(self,X,y):
        n, d = X.shape

        # Initial guess
        self.w = np.zeros((d, 1))

        # check the gradient
        estimated_gradient = approx_fprime(self.w, lambda w: self.funObj(w,X,y)[0], epsilon=1e-6)
        implemented_gradient = self.funObj(self.w,X,y)[1]
        if np.max(np.abs(estimated_gradient - implemented_gradient) > 1e-4):
            print('User and numerical derivatives differ: %s vs. %s' % (estimated_gradient, implemented_gradient));
        else:
            print('User and numerical derivatives agree.')

        self.w, f = findMin(self.funObj, self.w, 100, X, y)

    def funObj(self,w,X,y):

        ''' MODIFY THIS CODE '''
        # Calculate the function value
        # f = sum log(exp(w^T(xi-yi) + exp(yi - w^T(xi))
        f = np.sum(np.log(np.exp(X.dot(w) - y) + np.exp(y - X.dot(w))))

        # Calculate the gradient value
        g = X.T.dot((np.exp(X.dot(w) - y) - np.exp(y - X.dot(w))) / (np.exp(X.dot(w) - y) + np.exp(y - X.dot(w))))

        return (f,g)


# Least Squares with a bias added
class LeastSquaresBias:

    def fit(self,X,y):
        rows, cols = X.shape
        N = rows
        beta = np.ones((N, 1))
        Z = np.concatenate((X, beta), axis=1)

        a = np.dot(Z.T, Z)
        b = np.dot(Z.T, y)
        self.w = solve(a, b)

    def predict(self, X):
        ''' YOUR CODE HERE '''

        row, col = X.shape
        beta = np.ones((row, 1))
        Z = np.concatenate((X, beta), axis=1)

        yhat = np.dot(Z, self.w)

        return yhat

# Least Squares with polynomial basis
class LeastSquaresPoly:
    def __init__(self, p):
        self.leastSquares = LeastSquares()
        self.p = p

    def fit(self,X,y):
        Z = self.__polyBasis(X)
        a = np.dot(Z.T, Z)
        b = np.dot(Z.T, y)
        self.w = solve(a, b)

    def predict(self, X):
        Z1 = self.__polyBasis(X)
        yhat = np.dot(Z1, self.w)
        return yhat

    # A private helper function to transform any matrix X into
    # the polynomial basis defined by this class at initialization
    # Returns the matrix Z that is the polynomial basis of X.
    def __polyBasis(self, X):
        n = X.shape[0]
        d = self.p + 1
        Z = np.ones((n, d))

        for i in range(1, d):
            for j in range(0, n):
                Z[j][i] = X[j][0] ** i

        return Z

# Least Squares with RBF Kernel
class LeastSquaresRBF:
    def __init__(self, sigma):
        self.sigma = sigma

    def fit(self,X,y):
        self.X = X
        n, d = X.shape

        Z = self.__rbfBasis(X, X, self.sigma)

        # Solve least squares problem
        a = Z.T@Z + 1e-12*np.identity(n) # tiny bit of regularization
        b = Z.T@y
        self.w = solve(a,b)

    def predict(self, Xtest):
        Z = self.__rbfBasis(Xtest, self.X, self.sigma)
        yhat = Z@self.w
        return yhat

    def __rbfBasis(self, X1, X2, sigma):
        n1 = X1.shape[0]
        n2 = X2.shape[0]
        d = X1.shape[1]
        den = 1 / np.sqrt(2 * np.pi * (sigma** 2))

        D = (X1**2)@np.ones((d, n2)) + \
            np.ones((n1, d))@(X2.T** 2) - \
            2 * (X1@X2.T)

        Z = den * np.exp(-1* D / (2 * (sigma**2)))
        return Z
