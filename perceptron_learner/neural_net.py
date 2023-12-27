import numpy as np
from perceptron import *
from nn_utils import *


class NeuralNet:
    def __init__(
        self,
        depth=2,
        width=3,
        loss_func=mean_squared_loss,
        hidden_activation='sigmoid',
        output_activation='sigmoid',
        input_len=2
    ):
        self.input_len = input_len
        self.depth = depth
        self.width = width
        self.loss_func = loss_func
        self.hidden_activation = activationFunctions[hidden_activation]
        self.output_activation = activationFunctions[output_activation]

        # (input layer) | (hidden1, hidden2, ..., hiddenN) | (output layer)
        #   => numLayers = N + 2
        self.numLayers = self.depth + 1
        self.layers = [[None]] * self.depth + [[None]]
        self.layer_outputs = np.array(
            [[0] * self.input_len] + [[0] * self.width] * self.numLayers)

        # initialize layers with random weights
        self.initializeRandomWeights()

        # isolate output layer
        self.output_layer = self.layers[-1]

    def initializeRandomWeights(self):
        """
        Initialize random weights for each perceptron in the network.
        """
        # hidden layers
        for layerNum in range(self.numLayers - 1):
            prevLayerLen = None
            if layerNum == 0:
                prevLayerLen = len(self.input_len)
            else:
                prevLayerLen = len(self.layers[layerNum - 1])
            self.layers[layerNum] = [Perceptron(
                inputLen=prevLayerLen, weights=np.random.standard_normal(prevLayerLen + 1), activation=self.hidden_activation) for _ in range(self.width)]

        # output layer
        lastHiddenLen = len(self.layers[-2])
        self.layers[-1] = [Perceptron(
            inputLen=lastHiddenLen, weights=np.random.standard_normal(lastHiddenLen + 1), activation=self.output_activation) for _ in range(1)]

    def feedForward(self, inputs):
        """
        Compute the output of each node in the network by feeding the inputs forward
        through the network.
        """
        # output = []
        # output.append(inputs)

        for layerNum in range(self.numLayers):
            layerOutput = []
            for perceptron in self.layers[layerNum]:
                layerOutput.append(perceptron.computeOutput(output[-1]))
            self.layer_outputs[layerNum + 1] = layerOutput

        # self.layer_outputs = output

    def getLayerMatrix(self, layer):
        """Compute the matrix of weights for each layer:
                L_k = [

                        k_w_11    k_w_12    .   .   .

                        k_w_21    k_w_22    .   .   .

                        k_w_31    k_w_32    .   .   .

                        .       .   .

                        .       .       .

                        .       .            .
                    ]

            for layer k.
        """
        lMat = np.empty((1, 1))
        for neuronNum in range(len(layer)):
            np.append(lMat, layer[neuronNum].weights)
        return lMat

    def getLayerActGrad(self, layer, prevOuts):
        """ Computes a vector of each node's activation derivative in a given layer, based on the
            previous layer's outputs:

                nambla_g = [g_k1'(prevOuts), ..., g_kn'(prevOuts)]

            for layer nodes 1, ..., n in layer k.
        """
        actGrad = np.empty((1, 1))
        for neuronNum in range(len(layer)):
            np.append(actGrad, layer[neuronNum].activationGrad(prevOuts))
        return actGrad

    # TODO: one iteration of gradient descent
    def single_mb_gradientDescent(
        self,
        X,
        Y,
        alpha,
        momentum
    ):
        """
        Perform one iteration of mini-batch gradient descent.

        minibatch : list of (input, output) tuples
            input : list of floats
            output : float
        alpha : float
            learning rate
        momentum : float
            momentum coefficient
        """

        # TODO: fix accessing and updating weights

        delta = list()  # Empty list to store derivatives
        # stores weight updates
        delta_w = [0 for _ in range(len(self.layers))]
        # stores bias updates
        delta_b = [0 for _ in range(len(self.layers))]
        # Calculate the the error at output layer.
        error_o = (self.layer_outputs[-1] - Y)
        for i in reversed(range(len(self.layers) - 1)):
            # mutliply error with weights transpose to get gradients

            # TODO: fix how the weights are accessed here
            prevOuts = self.layer_outputs[i - 1]
            layerMat = self.getLayerMatrix(self.layers[i])
            # compute vector of all local activation gradients
            layerActivationGrad = self.getLayerActGrad(
                self.layers[i], prevOuts)
            error_i = np.multiply(
                layerMat.T.dot(error_o), layerActivationGrad)
            # store gradient for weights
            if self.loss_func == squared_loss:  # multiply  1/n by the residual sum of squares
                delta_w[i+1] = error_o.dot(self.layer_outputs[i].T)/len(Y)
            # store gradients for biases
            delta_b[i+1] = (1/len(Y)) * np.sum(error_o, axis=1,
                                               keepdims=True)
            # assign the previous layers error as current error and repeat the process.
            error_o = error_i

        delta_w[0] = error_o.dot(X)  # gradients for last layer
        if self.loss_func == squared_loss:  # multiply  1/n by the residual sum of squares
            delta_b[0] = np.sum(error_o, axis=1, keepdims=True)/len(Y)

        return (delta_w, delta_b)

    def update_layer_weights(self, layer, deltas_w, deltas_b, alpha):
        for neuron in layer:
            neuron.updateWeights(deltas_w, deltas_b, alpha)

    # TODO: training loop

    def mb_train(
        self,
        examples,
        alpha=0.1,
        numEpochs=100,
        momentum=0.9,
        decay=0.5,
        miniBatchSize=10
    ):
        """
        Loope over the training set for a specified number of epochs, performing
        mini-batch gradient descent on each epoch with decayed learning rate and
        momentum.

        examples : list of (input, output) tuples
            input : list of floats
            output : float
        alpha : float
            learning rate
        numEpochs : int
            number of iterations of gradient descent to perform

        """
        # TODO: implement mini-batching and momentum
        currAlpha = alpha
        iters = 0
        miniBatch = None
        while iters <= numEpochs:
            for X, Y in miniBatch:
                self.feedForward(X)
                dW, dB = self.single_mb_gradientDescent(
                    X, Y, momentum)
                for i in range(self.numLayers):
                    self.update_layer_weights(
                        self.layers[i], dW, dB, currAlpha)
            currAlpha *= decay
            iters += 1
