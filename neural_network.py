import numpy as np


class NeuralNetwork:
    def __init__(self, X, hidden, Y, learning_rate=0.0001, iterations=100):
        """Create a simple feedforward neural network with a single hidden layer

        Args:
            X (np.array(float)): Training data, of shape (input_unit, N)
            hidden (int): number of hidden units
            Y (np.array(float)): Training labels, of size (N, output_unit)
            learning_rate (float, optional): learning rate for gradient descent. Defaults to 0.01.
             iterations (int): Number of gradient descent steps
        """
        self.X = X
        self.input_unit, self.N = X.shape
        self.Y = Y
        self.hidden_unit = hidden
        self.output_unit = Y.shape[1]
        self.learning_rate = learning_rate
        self.iterations = iterations

        self.Theta_1, self.Theta_2 = self.initialize_parameters()

    def initialize_parameters(self):
        """Initialize the parameters of the neural network. Uses a N(0, 0.1^2) initialization."""

        return 0.1 * np.random.randn(self.input_unit + 1, self.hidden_unit), \
            0.1 * np.random.randn(self.hidden_unit + 1, self.output_unit)

    @staticmethod
    def tanh(a):
        """Returns tanh

        Args:
            a (number): input

        Returns:
            float: tanh(a)
        """
        return np.tanh(a)

    @staticmethod
    def tanhprime(a):
        """Returns derivative of tanh

        Args:
            a (number): input

        Returns:
            float: tanh'(a)
        """
        return 1 - NeuralNetwork.tanh(a)**2

    @staticmethod
    def linear(b, slope=1):
        """Returns a linear function y = slope*b

        Args:
            b (number): input
            slope (int, optional): slope of linear function. Defaults to 1.

        Returns:
            float: linear(b)
        """
        return slope * b

    @staticmethod
    def linearprime(c, slope=1):
        """Returns the derivative of the linear function y = slope*b

        Args:
            c (number): input
            slope (int, optional): slope of linear function. Defaults to 1.

        Returns:
            float: linear'(c)
        """
        return slope

    def forward_propagation(self, X):
        """Performs a forward propagation step of the NN

        Returns:
            np.array(float): the results from the feedforward NN, of shape (N, output_unit)
            dict: cache of intermediate values for backpropagation step
        """

        N = X.shape[1]
        outputs = np.zeros((N, self.output_unit))
        cache = []

        for n in range(N):
            Y0 = np.insert(X[:, n], 0, 1, axis=0)
            Z1 = self.Theta_1.T @ Y0
            A1 = NeuralNetwork.tanh(Z1)
            Y1 = np.insert(A1, 0, 1, axis=0)
            Z2 = self.Theta_2.T @ Y1
            A2 = NeuralNetwork.linear(Z2)
            # cache[n] stores the intermediate values of the forward prop of the n-th piece of training data
            cache.append({'Y0': Y0, 'Z1': Z1, 'A1': A1,
                         'Y1': Y1, 'Z2': Z2, 'A2': A2})
            outputs[n] = A2

        return outputs, cache

    def loss(self, cache):
        """Returns average loss of training data with current weights

        Args:
            cache (list): Cache of values from the forward prop step

        Returns:
            (float): Average difference of squares loss
        """
        losses = np.zeros(self.N)
        for n in range(self.N):
            losses[n] = 0.5 * np.sum((cache[n]['A2'] - self.Y[n]) ** 2)

        return np.sum(losses) / losses.size

    def backward_propagation(self, cache):
        """Performs a backpropagation step of the NN

        Args:
            cache (list): Cache of values from the forward prop step

        Returns:
            (np.array(float), np.array(float)): the gradients for `Theta_1`, `Theta_2`
        """

        # Y used to be shape (1, N)
        # I changed Y to shape (N, output_unit) so that Y[n] is a vector label of the n-th training data point

        # Delta_2[n][j] is the derivative of loss function J_n w.r.t. the linear combiner of the j-th neuron of the
        # final layer for the n-th piece of training data, 0-indexed
        Delta_2 = np.zeros((self.N, self.output_unit))

        for n in range(self.N):
            # See (18.24) p. 890
            Delta_2[n] += (cache[n]['A2'] - self.Y[n]) * self.linearprime(cache[n]['Z2'])

        # Delta_1[n][j] is the derivative of loss function J_n w.r.t. the linear combiner of the j-th neuron of the
        # middle layer for the n-th piece of training data
        Delta_1 = np.zeros((self.N, self.hidden_unit))
        E = np.zeros((self.N, self.hidden_unit))
        for n in range(self.N):
            for j in range(self.hidden_unit):
                # See (18.31) p. 891
                # j+1 because first row of Theta_2 is for biases. Theta_2[j+1] corr. to the weights to the output layer
                # from the hidden layer's j-th neuron.
                # TODO numpy magic here
                E[n][j] += np.dot(Delta_2[n], self.Theta_2[j+1])
                # See (18.32) p. 891
                Delta_1[n, j] += E[n, j] * self.tanhprime(cache[n]['Z1'][j])

        # Grad_i has the gradients for Theta_i

        Grad_1 = np.zeros((self.input_unit + 1, self.hidden_unit))
        Grad_2 = np.zeros((self.hidden_unit + 1, self.output_unit))

        # Grad_2[i+1][j] is the gradient of the link between the i-th neuron of the hidden layer and the j-th neuron of
        # the output layer, while Grad_2[0][j] is the gradient of the bias of the j-th neuron of the output layer
        for i in range(self.hidden_unit + 1):
            for n in range(self.N):
                # TODO sum this better
                # See (18.21) p. 890
                Grad_2[i, :] += Delta_2[n, :] * cache[n]['Y1'][i]

        # Grad_1[i+1][j] is the gradient of the link between the i-th neuron of the input layer and the j-th neuron of
        # the hidden layer, while Grad_1[0][j] is the gradient of the bias of the j-th neuron of the hidden layer
        for i in range(self.input_unit + 1):
            for n in range(self.N):
                # See (18.21) p. 890
                Grad_1[i, :] += Delta_1[n, :] * cache[n]['Y0'][i]
        return Grad_1, Grad_2

    def gradient_descent(self, grads):
        """Performs gradient descent, given a batch of gradients

        Args:
            grads (np.array(float), np.array(float)): gradients from backpropagation
        """
        self.Theta_1 = self.Theta_1 - self.learning_rate * grads[0]
        self.Theta_2 = self.Theta_2 - self.learning_rate * grads[1]

    def train(self, X):
        """Trains a neural network on the training data stored in X and Y

        Returns:
            (NeuralNetwork): A trained neural network
        """

        for j in range(self.iterations):
            cache = self.forward_propagation(X)[1]
            grads = self.backward_propagation(cache)
            self.gradient_descent(grads)
            print("Loss before iteration", j+1, ":", self.loss(cache))

        return self
