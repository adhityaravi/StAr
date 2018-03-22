import numpy as np

class CrossValidation():
    """Performs different types of cross validation on the given dataset using the specified classifier

        Should be initialized with an object pointing to a classifier and the dataset for validation

        For example:
        clf = DecisionTreeClassifier()
        dataset =  [['Green', 3, 'Apple'],
                    ['Yellow', 3, 'Apple'],
                    ['Red', 1, 'Grape'],
                    ['Red', 1, 'Grape'],
                    ['Yellow', 3, 'Lemon']]
        CV = CrossValidation(dataset, clf)

        Cross validation of desired type can be then performed. For example:
        accuracy = CV.kFoldValidation(kSample=5)

        Note: Only k-fold available for now"""

    def __init__(self, dataSet, clf):
        self.dataSet = dataSet
        # --> Shuffling the data to improve randomness
        np.random.shuffle(self.dataSet)
        self.clf = clf

    def predictionAccuracy(self, predictions, actualValues):
        """Calculates the accuracy to which the predictions are correct"""

        correctPredictions = np.sum(predictions == actualValues)
        totalPredictions = len(predictions)

        # --> Accuracy = number of correct predictions made by the classifier / total number of predictions
        accuracy = float(correctPredictions) / float(totalPredictions)

        return accuracy

    def sampling(self, dataSet, sampleSize):
        """Performs sampling of the given data and of given size

            Note: Sampling is not stratified. Hence the sample distribution may not follow the original distribution
            ToDo: Use stratified sampling to maintain the original data distribution"""

        # --> Random sampling from the dataset
        sampleID = np.random.choice(dataSet.shape[0], sampleSize, replace=False)
        sample = dataSet[sampleID]

        return sample, sampleID

    def kFoldTestTrainSplit(self, kFoldSamples, sampleID1):
        """Creates a training and testing dataset from the k-fold samples

            Uses one of the k-fold samples as the test data and the other samples
            are stacked to form the training data"""

        testData = kFoldSamples[sampleID1, :, :]
        trainData = kFoldSamples[sampleID1-1, :, :]

        if sampleID1-1 is -1:
            sampleID2 = kFoldSamples.shape[0] - 1
        else:
            sampleID2 = sampleID1 - 1

        # --> Stacking samples to create a training dataset
        for iSample in range(kFoldSamples.shape[2]):
            if iSample is not sampleID1 and iSample is not sampleID2:
                trainData = np.append(trainData, kFoldSamples[iSample, :, :], axis=0)

        return trainData, testData

    def kFoldValidation(self, kSample=5):
        """Performs k-fold cross validation with default k value as 5"""

        # --> Calculating the size of a single k-fold sample
        nSize = self.dataSet.shape[0]
        kSize = int(nSize/ kSample)

        # --> Initializing the k-fold samples
        kFoldSamples = np.zeros([kSample, kSize, self.dataSet.shape[1]], order='F')
        iSample = 0

        accuracy = 0

        # --> Sampling k datasets from the given dataset
        while kSize <= nSize:

            kFoldSamples[iSample, :, :], sampleID = self.sampling(self.dataSet, kSize)

            self.dataSet = np.delete(self.dataSet, sampleID, axis=0)

            nSize = self.dataSet.shape[0]

            iSample += 1

        # --> Performing k-fold validation
        for iSample in range(kSample):

            # --> Predictions using given classifier
            trainData, testData = self.kFoldTestTrainSplit(kFoldSamples, iSample)
            self.clf.fit(trainData)
            predictions = self.clf.predict(testData)

            # --> Calculating the accuracy of the predictions
            accuracy += self.predictionAccuracy(predictions, testData[:, -1])

        # --> Calculating mean accuracy
        meanAccuracy = accuracy / kSample

        return meanAccuracy

    def holdOutValidation(self, testSize=0.3):
        """Performs cross validation using hold out method with default test train ratio as 0.3"""

        # --> Calculating the size of test dataset
        nSize = self.dataSet.shape[0]
        nTest = int(testSize * nSize)

        # --> Sampling testing data from the dataset
        testData, sampleID = self.sampling(self.dataSet, nTest)

        # --> Extracting training data from the dataset
        trainData = np.delete(self.dataSet, sampleID, axis=0)

        # --> Prediction using given classifier
        self.clf.fit(trainData)
        predictions = self.clf.predict(testData)

        # --> Calculating accuracy of the predictions
        accuracy = self.predictionAccuracy(predictions, testData[:, -1])

        return accuracy











