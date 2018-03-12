# standard Python imports
import os
import argparse
import time
import pickle

# 3rd party libraries
import numpy as np                              # this comes with Anaconda
import matplotlib.pyplot as plt                 # this comes with Anaconda
import pandas as pd                             # this comes with Anaconda
from sklearn.tree import DecisionTreeClassifier # see http://scikit-learn.org/stable/install.html
from sklearn.neighbors import KNeighborsClassifier # same as above

# CPSC 340 code
import utils
from decision_stump import DecisionStump, DecisionStumpEquality, DecisionStumpInfoGain
from decision_tree import DecisionTree
from knn import KNN, CNN

def load_dataset(filename):
    with open(os.path.join('..','data',filename), 'rb') as f:
        return pickle.load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--question', required=True,
        choices=["1.1", "2", "2.2", "2.3", "2.4", "3", "3.1", "3.2", "4.1", "4.2", "5"])

    io_args = parser.parse_args()
    question = io_args.question

    if question == "1.1":

        # retrieve max, min, median, mode
        ds = pd.read_csv('../data/fluTrends.csv')

        maximum = ds.values.max()
        minimum = ds.values.min()
        median = ds.stack().median()
        mode = utils.mode(ds.values)

        results = [maximum, minimum, median, mode]

        # retrieve quantiles

        print("quantiles: %s" % ds.stack().quantile([0.05, 0.25, 0.5, 0.75, 0.95]))

        # retrieve maximum mean, minimum mean, highest variance, lowest variance

        means = ds.mean()
        variances = ds.var()

        maxMean = means.idxmax(axis=0)
        minMean = means.idxmin(axis=0)
        maxVar = variances.idxmax(axis=0)
        minVar = variances.idxmin(axis=0)

        print(maxMean, minMean, maxVar, minVar)

    elif question == "2":

        # 1. Load citiesSmall dataset
        dataset = load_dataset("citiesSmall.pkl")
        X = dataset["X"]
        y = dataset["y"]

        # 2. Evaluate majority predictor model
        y_pred = np.zeros(y.size) + utils.mode(y)

        error = np.mean(y_pred != y)
        print("Mode predictor error: %.3f" % error)

        # 3. Evaluate decision stump
        model = DecisionStumpEquality()
        model.fit(X, y)
        y_pred = model.predict(X)

        error = np.mean(y_pred != y) 
        print("Decision Stump with inequality rule error: %.3f" % error)

        # PLOT RESULT
        utils.plotClassifier(model, X, y)

        fname = os.path.join("..", "figs", "q2_decisionBoundary.pdf")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)


    elif question == "2.2":
        # 1. Load citiesSmall dataset
        dataset = load_dataset("citiesSmall.pkl")
        X = dataset["X"]
        y = dataset["y"]

        # 3. Evaluate decision stump
        model = DecisionStump()
        model.fit(X, y)
        y_pred = model.predict(X)

        error = np.mean(y_pred != y)
        print("Decision Stump with inequality rule error: %.3f" % error)

        # PLOT RESULT
        utils.plotClassifier(model, X, y)

        fname = os.path.join("..", "figs", "q2_2_decision_plot_classifier.pdf")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)

    elif question == "2.3":
        # 1. Load citiesSmall dataset
        dataset = load_dataset("citiesSmall.pkl")
        X = dataset["X"]
        y = dataset["y"]

        # 2. Evaluate decision tree
        model = DecisionTree(max_depth=2)
        model.fit(X, y)

        y_pred = model.predict(X)
        error = np.mean(y_pred != y)

        print("Error: %.3f" % error)
    
    elif question == "2.4":
        dataset = load_dataset("citiesSmall.pkl")
        X = dataset["X"]
        y = dataset["y"]
        print("n = %d" % X.shape[0])

        depths = np.arange(1,15) # depths to try

        t = time.time()
        my_tree_errors = np.zeros(depths.size)
        for i, max_depth in enumerate(depths):
            model = DecisionTree(max_depth=max_depth)
            model.fit(X, y)
            y_pred = model.predict(X)
            my_tree_errors[i] = np.mean(y_pred != y)
        print("Our decision tree took %f seconds" % (time.time()-t))
        
        plt.plot(depths, my_tree_errors, label="my implementation")
        
        t = time.time()
        sklearn_tree_errors = np.zeros(depths.size)
        for i, max_depth in enumerate(depths):
            model = DecisionTreeClassifier(max_depth=max_depth, criterion='entropy', random_state=1)
            model.fit(X, y)
            y_pred = model.predict(X)
            my_tree_errors[i] = np.mean(y_pred != y)
        print("scikit-learn's decision tree took %f seconds" % (time.time()-t))

        plt.plot(depths, my_tree_errors, label="sklearn")
        plt.xlabel("Depth of tree")
        plt.ylabel("Classification error")
        plt.legend()
        fname = os.path.join("..", "figs", "q2_4_tree_errors.pdf")
        plt.savefig(fname)
        
        tree = DecisionTreeClassifier(max_depth=1)
        tree.fit(X, y)

        y_pred = tree.predict(X)
        error = np.mean(y_pred != y)

        print("Error: %.3f" % error)

    elif question == "3":
        dataset = load_dataset("citiesSmall.pkl")
        X, y = dataset["X"], dataset["y"]
        X_test, y_test = dataset["Xtest"], dataset["ytest"]        
        model = DecisionTreeClassifier(max_depth=2, criterion='entropy', random_state=1)
        model.fit(X, y)

        y_pred = model.predict(X)
        tr_error = np.mean(y_pred != y)

        y_pred = model.predict(X_test)
        te_error = np.mean(y_pred != y_test)
        print("Training error: %.3f" % tr_error)
        print("Testing error: %.3f" % te_error)

    elif question == "3.1":
        # load in the data set
        dataset = load_dataset("citiesSmall.pkl")
        X, y = dataset["X"], dataset["y"]
        X_test, y_test = dataset["Xtest"], dataset["ytest"]

        depths = np.arange(1, 15)

        training_errors = np.zeros(depths.size)
        test_errors = np.zeros(depths.size)

        for i, max_depth in enumerate(depths):
            model = DecisionTreeClassifier(criterion='entropy', max_depth=max_depth, random_state=1)
            model.fit(X, y)

            y_pred = model.predict(X)
            tr_error = np.mean(y_pred != y)
            training_errors[i] = tr_error

            y_pred = model.predict(X_test)
            te_error = np.mean(y_pred != y_test)
            test_errors[i] = te_error

        plt.plot(depths, training_errors, label="Training Error")
        plt.plot(depths, test_errors, label="Test Error")
        plt.xlabel("Depth of tree")
        plt.ylabel("Classification error")
        plt.legend()
        fname = os.path.join("..", "figs", "q3_1_train_vs_test.pdf")
        plt.savefig(fname)

    elif question == "3.2":
        # load in the data set
        dataset = load_dataset("citiesSmall.pkl")
        X, y = dataset["X"], dataset["y"]
        n, d = X.shape

        x_training = X[:n//2]
        y_training = y[:n//2]
        x_validation = X[n//2:]
        y_validation = y[n//2:]

        swap = 0
        if swap:
            x_training, x_validation = x_validation, x_training
            y_training, y_validation = y_validation, y_training

        depths = np.arange(1, 15)

        training_errors = np.zeros(depths.size)
        validation_errors = np.zeros(depths.size)

        for i, max_depth in enumerate(depths):

            model = DecisionTreeClassifier(max_depth=max_depth, criterion='entropy', random_state=1)
            model.fit(x_training, y_training)

            y_pred = model.predict(x_training)
            tr_error = np.mean(y_pred != y_training)

            y_pred = model.predict(x_validation)
            val_error = np.mean(y_pred != y_validation)

            training_errors[i] = tr_error
            validation_errors[i] = val_error

        plt.plot(depths, training_errors, label="training error")
        plt.plot(depths, validation_errors, label="testing error")
        plt.xlabel("Depth")
        plt.ylabel("Error")
        plt.legend()

        fname = os.path.join("..", "figs", "q3_2_train_test.pdf")
        plt.savefig(fname)

        print("Best depth:", depths[np.argmin(validation_errors)])

    if question == '4.1':
        dataset = load_dataset('citiesSmall.pkl')
        X = dataset['X']
        y = dataset['y']
        X_test = dataset['Xtest']
        y_test = dataset['ytest']

        for k in [1, 3, 10]:
            model = KNN(k)
            model.fit(X, y)
            training_error = np.mean(y != model.predict(X))
            test_error = np.mean(y_test != model.predict(X_test))
            print('k={:2d}\ttrainErr: {:.3f}%\ttestErr: {:.3f}%'.format(k, training_error, test_error))

        # generate the plots
        model = KNN(1)
        model.fit(X, y)
        utils.plotClassifier(model, X, y)
        fname = os.path.join("..", "figs", "q4_1_knn_plot_classifier.pdf")
        plt.savefig(fname)

        model = KNeighborsClassifier(1)
        model.fit(X,y)
        utils.plotClassifier(model, X, y)
        fname = os.path.join("..", "figs", "q4_1_knn_plot_classifier_learn.pdf")
        plt.savefig(fname)

    if question == '4.2':

        # 1 & part 2
        t = time.time()
        dataset = load_dataset('citiesBig1.pkl')
        X = dataset['X']
        y = dataset['y']
        X_test = dataset['Xtest']
        y_test = dataset['ytest']

        model = CNN(1)
        model.fit(X, y)

        training_error = np.mean(y != model.predict(X))
        test_error = np.mean(y_test != model.predict(X_test))
        print("CNN took %f seconds" % (time.time()-t))
        print(training_error, test_error)

        # part 3
        utils.plotClassifier(model, X, y)
        fname = os.path.join("..", "figs", "q4_2_cnn_plot_classifier.pdf")
        plt.savefig(fname)

        # part 6
        dataset = load_dataset('citiesBig2.pkl')
        with open('../data/citiesBig2.pkl', 'rb') as f:
            data = pickle.load(f)
        X = dataset['X']
        y = dataset['y']
        X_test = dataset['Xtest']
        y_test = dataset['ytest']

        model = CNN(1)
        model.fit(X, y)

        training_error = np.mean(y != model.predict(X))
        test_error = np.mean(y_test != model.predict(X_test))
        print(training_error, test_error)

        # part 7
        dataset = load_dataset('citiesBig1.pkl')
        X = dataset['X']
        y = dataset['y']

        model = DecisionTreeClassifier()
        model.fit(X, y)

        utils.plotClassifier(model, X, y)
        fname = os.path.join("..", "figs", "q4_2_decision_plot_classifier.pdf")
        plt.savefig(fname)
