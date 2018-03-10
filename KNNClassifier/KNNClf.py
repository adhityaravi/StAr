from scipy.spatial import distance

class KNNClassifier():
    """Bare bones K nearest neighbour classifier with k=1"""

    def fit(self, x_train, y_train):
        """Trains the classifier with the input data"""

        self.x_train = x_train
        self.y_train = y_train

    def predict(self, x_test):
        """Predicts an output depending on the training data"""

        predictions=[]

        for i_test in x_test:
            label = self.closest(i_test)
            predictions.append(label)

        return predictions

    def closest(self, i_test):
        """Finds out the closest point in the training data to the given test data"""

        best_dist = self.euc_dist(i_test, self.x_train[0])
        best_index = 0

        for i in range(len(self.x_train)):
            dist = self.euc_dist(i_test, self.x_train[i])

            if dist < best_dist:
                best_dist = dist
                best_index = i

        return self.y_train[best_index]

    def euc_dist(self, a, b):
        """Calculates the Euclidean distance between a and b"""

        dist = distance.euclidean(a, b)

        return dist