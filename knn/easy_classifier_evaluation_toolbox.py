from __future__ import division
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
import numpy as np
import copy

class knnClassifier:

    def __init__(self, X, y, n):
        # X = N*D, y = N*1
        self.X = X
        self.y = y
        self.knn = KNeighborsClassifier(n)
        self.knn_result = []

    def train():
        self.knn.fit(X, y)

    def predict(self, X_test):
        self.knn_result = self.knn.predict(X_test)
        return self.knn_result

    def getAccuracy(self, y_test):
        # y_test must be an numpy array with shape of N(test size)*1
        intersection = len(y_test)-np.count_nonzero(self.knn_result-y_test)
        return intersection/(len(y_test))

    def getF1(self, y_test):
        # y_test must be an numpy array with shape of N(test size)*1
        return f1_score(y_test, self.knn_result, average='macro')

class knnRecommender:

    def __init__(self, X_train, labels):
        self.X_train = X_train
        self.y_train = labels
        self.recommendation = np.array([])

    def compute_distances(self, X):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        inner = 2*np.dot(X, np.transpose(self.X_train))
        test_norm = np.sum(np.square(X), axis = 1)
        train_norm = np.sum(np.square(self.X_train), axis = 1)
        test_norm = np.expand_dims(test_norm, axis = 1)
        train_norm = np.expand_dims(train_norm, axis = 0)
        test_norm = np.tile(test_norm, num_train)
        train_norm = np.repeat(train_norm, num_test, axis = 0)
        dists = (-inner+test_norm+train_norm)**0.5
        return dists

    def recommend(self, x_test, k=1):
        # Here x_test is N(test)*D, which is a single image
        # Return the index of X_train
        dists = self.compute_distances(x_test)
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        infinity = float("inf")

        recommendation = np.zeros((k, num_test))
        for i in xrange(k):
            curr_min = np.argmin(dists, axis = 1)
            recommendation[i,:] = curr_min
            dists[range(dists.shape[0]),curr_min] = infinity
        self.recommendation = recommendation
        return recommendation

    def get_precision(self, recommend, y_test):
        k = recommend.shape[0]
        rec_label = copy.deepcopy(recommend)
        for i in xrange(k):
            rec_label[i] = self.y_train[rec_label[i].astype(int)]
        result = rec_label-y_test
        non_zero = np.count_nonzero(result, axis = 0)
        precision = (k-non_zero)/k
        return precision
