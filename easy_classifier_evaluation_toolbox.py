from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
import numpy as np

class knnClassifier:

    def __init__(self, X, y):
        # X = N*D, y = N*1
        self.X = X
        self.y = y
        self.knn = KNeighborsClassifier()
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
