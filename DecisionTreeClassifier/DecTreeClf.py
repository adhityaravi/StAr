from  __future__ import print_function

class DecisionTreeClassifier():
    """Decision Tree Classifier built based on CART algorithm"""

    def __init__(self, dataSet = None, header = None):
        self.dataSet = dataSet
        self.header = header

    def fetchUniqueVal(self, dataSet, featureID):
        """Determines the unique data values of a feature"""

        return set([data[featureID] for data in dataSet])

    def countLabel(self, dataSet):
        """Counts the number of labels in the given dataset"""

        # --> Initializing Label_Counts dictionary
        labelCounts = {}

        for data in dataSet:
            # --> Fetching the label from the each data in the dataset
            label = data[-1]

            # --> Initializing dictionary for new label
            if label not in labelCounts:
                labelCounts[label] = 0

            labelCounts[label] += 1

        return labelCounts

    def isNumeric(self, value):
        """Determines if value is a numeric value or not"""

        return isinstance(value, int) or isinstance(value, float)

    def partitionDataSet(self, dataSet, question):
        """Partitions the DataSet based on the given Question"""

        truePart, falsePart = [], []

        # --> Looping through every entry in the dataset and categorizing them
        for Data in dataSet:
            if question.matchFeature(Data):
                truePart.append(Data)
            else:
                falsePart.append(Data)

        return truePart, falsePart

    def calcGiniImpurity(self, dataSet):
        """Calculates the Gini Impurity of the given DataSet"""

        labelCounts = self.countLabel(dataSet)

        # --> Initializing impurity
        impurity = 1

        for label in labelCounts:
            probOfLabel = labelCounts[label] / float(len(dataSet))
            impurity -= probOfLabel ** 2

        return impurity

    def calcInfoGain(self, truePart, falsePart, currentUncertainity):
        """Calculates the information gain because of asking a question to the dataset"""

        weight = float(len(truePart)) / (len(truePart) + len(falsePart))

        # --> Info gain = current uncertainity - sum of weighted impurities of true and false parts
        return currentUncertainity - weight*self.calcGiniImpurity(truePart) - (1-weight)*self.calcGiniImpurity(falsePart)

    def findBestQuestion(self, dataSet):
        """Finds the best question to ask to the given dataset so as to have maximum info gain"""

        bestInfoGain = 0
        bestQuestion = None

        # --> Gini impurity of the current given dataset
        currentUncertainity = self.calcGiniImpurity(dataSet)
        # --> Number of features in the dataset
        nFeature = len(dataSet[0]) - 1

        # --> Looping through the different features
        for iFeature in range(nFeature):

            # --> Fetching all the unique values of this particular feature
            nFeatureValue = self.fetchUniqueVal(dataSet, iFeature)

            # --> Looping through the different feature values
            for iFeatureValue in nFeatureValue:

                # --> Creating a question about the feature with a feature value
                question = Question(iFeature, iFeatureValue, self.header)

                # --> Partitioning the dataset based on the question
                truePart, falsePart = self.partitionDataSet(dataSet, question)

                # --> Check if the asked question creates a partition
                if len(truePart) == 0 or len(falsePart) == 0:
                    continue

                # --> Calculate the info gain because of the asked question
                infoGain = self.calcInfoGain(truePart, falsePart, currentUncertainity)

                # --> Comparison with the best gain so far
                if infoGain > bestInfoGain:
                    bestInfoGain, bestQuestion = infoGain, question

        return bestInfoGain, bestQuestion

    def buildTree(self, dataSet):
        """Builds the decision tree for the given dataset"""

        # --> Find the best question that can be asked and gain for the given dataset
        infoGain, question = self.findBestQuestion(dataSet)

        # --> Check if we've reached a Leaf node
        if infoGain == 0:
            return Leaf(dataSet)

        # --> If a Leaf is not reached, partition the dataset based on the question
        truePart, falsePart = self.partitionDataSet(dataSet, question)

        # --> Recursively build a branch for the true and false parts
        trueBranch = self.buildTree(truePart)
        falseBranch = self.buildTree(falsePart)

        # --> Return a Decision node with a best question that can be asked and the reference to the branches created
        # --> because of this question
        return DecisionNode(question, trueBranch, falseBranch)

    def printTree(self, tree, spacing=""):
        """Visualizes the decision tree that the classifier has built"""

        # --> Check if a Leaf node is reached
        if isinstance(tree, Leaf):
            print(spacing + "Predict", self.printLeaf(tree.predictions))
            return

        # --> If not print the question at the current node
        print(spacing + str(tree.question))

        # --> Recursive call to the true and flase branches created because of the question
        print(spacing + "--> True Branch:")
        self.printTree(tree.trueBranch, spacing+"  ")

        print(spacing + "--> False Branch:")
        self.printTree(tree.falseBranch, spacing+"  ")

    def printLeaf(self, labelCounts):
        """Prints the confidence data of the Leaf node"""

        total = sum(labelCounts.values()) * 1.0

        # --> Inititalizing confidence data
        conf = {}

        # --> Looping through the labels in the leaf node and calculating confidence for each label
        for label in labelCounts.keys():
            conf[label] = str(int(labelCounts[label]/total * 100)) + "%"

        return conf

    def classifyData(self, data, tree):
        """Predict the label for the given data using the built decision tree"""

        # --> Check if a leaf node is reached
        if isinstance(tree, Leaf):
            return tree.predictions

        # --> If not continue along the decision tree by deciding whether to follow the true branch or the false branch
        # --> until a leaf node is reached
        if tree.question.matchFeature(data):
            return self.classifyData(data, tree.trueBranch)
        else:
            return self.classifyData(data, tree.falseBranch)

    def predict(self, testData):
        """Build a decision tree and predict the labels for a test dataset"""

        # --> Building decision tree
        tree = self.buildTree(self.dataSet)

        # --> Inititalizing predictions
        predictions = []

        # Classifying test data
        for data in testData:
            predictions.append(self.classifyData(data, tree))

        return predictions

    def printPredictions(self, predictions):
        """Print the predictions made in a readable format"""

        for data in predictions:
            print("Predicted %s" % self.printLeaf(data))


class Question(DecisionTreeClassifier):
    """Question class is used to ask a question based on FeatureID and FeatureValue

        Based on the asked question, the dataset is partitioned into 2 parts. A true-part that satisfies the asked
        question and a false-part which does not satisfy the question asked"""

    def __init__(self, qFeatureID, qFeatureValue, header=None):

        DecisionTreeClassifier.__init__(self, None, header)
        self.qFeatureID = qFeatureID
        self.qFeatureValue = qFeatureValue

    def matchFeature(self, data):
        """Matches the Feature value in the given Data with same the Feature value in the question"""

        mFeatureValue = data[self.qFeatureID]

        # --> If its a numeric value it check for the target feature value being >= feature value in question or
        # --> for a non numeric value it checks for the target feature value being == feature value in question
        if self.isNumeric(self.qFeatureValue):
            return mFeatureValue >= self.qFeatureValue
        else:
            return mFeatureValue == self.qFeatureValue

    def __repr__(self):
        """Represent the question in a readable format if a header is available"""

        if self.header is not None:
            condition = "=="
            if self.isNumeric(self.qFeatureValue):
                condition = ">="

            return "Is %s %s %s?" % (self.header[self.qFeatureID], condition, str(self.qFeatureValue))

        else:
            return self.header

class Leaf(DecisionTreeClassifier):
    """Leaf class behaves like a node that is the destination of data after travelling through the decision tree

        Leaf node holds the dictionary that represents the different labels that has reached a destination in the tree
        and also the number of the labels that have reached this destination"""

    def __init__(self, dataSet):

        DecisionTreeClassifier.__init__(self)
        self.predictions = self.countLabel(dataSet)

class DecisionNode():
    """Decision node is where a particular question is asked to the dataset and the dataset is partitioned into true
        and false parts

        Decision node holds a reference to the question asked, the true branch and the false branch"""

    def __init__(self, question, trueBranch, falseBranch):
        self.question = question
        self.trueBranch = trueBranch
        self.falseBranch = falseBranch











