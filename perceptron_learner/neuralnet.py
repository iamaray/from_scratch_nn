import numpy as np
from perceptron import *
from nn_utils import *


class NeuralNet:
    def __init__(self, inputs, depth=2, width=3, loss_func='L2', hidden_activation='sigmoid', output_activation='sigmoid'):
        self.input_len = len(inputs)
        self.depth = depth
        self.width = width
        self.loss_func = lossFunctions[loss_func]
        self.hidden_activation = activationFunctions[hidden_activation]
        self.output_activation = activationFunctions[output_activation]

        # (input layer) | (hidden1, hidden2, ..., hiddenN) | (output layer)
        #   => numLayers = N + 2
        self.numLayers = self.depth + 2
        self.layers = [inputs] + [[]] * self.depth + [[]]
        self.layer_outputs = np.array([[]] * self.numLayers)
        self.output = None

        # initialize random weights for each perceptron in the network
        self.initializeRandomWeights()
        # compute the output of each node in the network by feeding the inputs forward
        self.feedForward()

    def initializeRandomWeights(self):
        """
        Initialize random weights for each perceptron in the network.
        """
        # hidden layers
        for layerNum in range(1, self.numLayers - 1):
            prevLayerLen = len(self.layers[layerNum - 1])
            self.layers[layerNum] = [Perceptron(
                inputLen=prevLayerLen, weights=np.random.sample((prevLayerLen,)), activation=self.hidden_activation) for _ in range(self.width)]

        # output layer
        prevLayerLen = len(self.layers[-2])
        self.layers[-1] = [Perceptron(
            inputLen=prevLayerLen, weights=np.random.sample((prevLayerLen,)), activation=self.output_activation) for _ in range(1)]

    def feedForward(self):
        """
        Compute the output of each node in the network by feeding the inputs forward
        through the network.
        """
        # input layer outputs = input layer inputs
        self.layer_outputs[0] = np.array(self.layers[0])

        for layerNum in range(1, self.numLayers):
            self.layer_outputs[layerNum] = [
                node.computeOutput(self.layer_outputs[layerNum - 1]) for node in self.layers[layerNum]]

        # output layer output
        self.output = self.layer_outputs[-1][0]
