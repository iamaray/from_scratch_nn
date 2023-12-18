import numpy as np
from nn_utils import *


class Perceptron:
    def __init__(self, inputLen=1, weights=None, activation='sigmoid'):
        # account for bias
        self.inputLen = inputLen + 1
        self.activationFunc = activationFunctions[activation]
        self.activationDeriv = activationFunctions[activation + '_deriv']

        if weights == None:
            self.weights = np.random.random_sample((inputLen, ))

        else:
            self.weights = weights

    def computeOutput(self, inputLst):
        """
        Calculate this node's output by applying the chosen activation function to
        a weighted sum over the inputs from the previous layer. In other words, we 
        are computing g(<W,I>), where W is the vector of weights, and I is the 
        vector of inputs from the previous layer.

        Args:
            inputLst (np.array(float)): a vector of all inputs fed into this node
                from the previous layer (without a bias term).

        Returns:
            float: the weighted input passed into the activation function.
        """
        return self.activationFunc(np.dot([1] + inputLst, self.weights))

    def updateWeights(self, deltas):
        """
        Update the weights of this perceptron using the delta rule.
        """
        for i in range(len(self.weights)):
            self.weights[i] += deltas[i]
