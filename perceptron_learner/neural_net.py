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
        self.layer_outputs = np.array([[0] * self.input_len] + [[0] * self.width] * self.numLayers)

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

    # TODO: one iteration of gradient descent
    def single_mb_gradientDescent(
        self,
        miniBatch,
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
        
        for X, Y in miniBatch:
            # perform a forward pass to compute the outputs at each layer
            self.feedForward(X)

            delta = list()  # Empty list to store derivatives
            # stores weight updates
            delta_w = [0 for _ in range(len(self.layers))]
            # stores bias updates
            delta_b = [0 for _ in range(len(self.layers))]
            # Calculate the the error at output layer.
            error_o = self.loss_func(self.layer_outputs[-1], Y)
            for i in reversed(range(len(self.layers) - 1)):
                # mutliply error with weights transpose to get gradients
                
                # TODO: fix how the weights are accessed here
                error_i = np.multiply(
                    self.layers[i+1].weights.T.dot(error_o), self.layers[i].activationGrad(self.layer_outputs[i - 1]))
                # store gradient for weights
                delta_w[i+1] = error_o.dot(self.layers[i].T)/len(y)
                # store gradients for biases
                delta_b[i+1] = np.sum(error_o, axis=1,
                                      keepdims=True)/len(y)
                # assign the previous layers error as current error and repeat the process.
                error_o = error_i
            delta_w[0] = error_o.dot(X)  # gradients for last layer
            
            if self.loss_func == squared_loss: # multiply  1/n by the residual sum of squares
                delta_b[0] = np.sum(error_o, axis=1, keepdims=True)/len(y)
            
            # return (delta_w, delta_b)
            
            # TODO: update weights

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
        while iters <= numEpochs:
            self.single_mb_gradientDescent(examples, currAlpha)
            currAlpha *= decay
            
            
