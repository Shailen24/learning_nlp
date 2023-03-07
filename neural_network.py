import numpy as np


class NeuralNetwork:
    def __init__(self, inp, hidden, output):
        """Create a simple feedforward neural network with a single hidden layer

        Args:
            inp (int): dimension of input
            hidden (int): number of hidden units
            output (int): dimension of output
        """
        self.parameters = self.initialize_parameters(inp, hidden, output)
        self.input_unit = inp
        self.hidden_unit = hidden
        self.output_unit = output

    def initialize_parameters(self, input_unit, hidden_unit, output_unit):
        """Initialize the parameters of the neural network.
        Uses a N(0, 0.1^2) initialization.

        Args:
            input_unit (int): dimension of input
            hidden_unit (int): number of hidden units
            output_unit (int): dimension of output
        """
        self.Theta_1 = 0.1 * np.random.randn(input_unit + 1, hidden_unit)
        self.Theta_2 = 0.1 * np.random.randn(hidden_unit + 1, output_unit)

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

        Args:
            X (np.array(float)): Input, of shape (input_unit, N)

        Returns:
            np.array(float): the results from the feedforward NN
            dict: cache of intermediate values for backpropagation step
        """
        outputs = np.zeros((X.shape[1], self.output_unit))
        cache = []
        N = X.shape[1]
        for n in range(N):
            Y0 = np.vstack((np.array([1]), X[:, n]))
            Z1 = self.Theta_1.T @ Y0
            A1 = NeuralNetwork.tanh(Z1)
            Y1 = np.vstack((np.array([1]), A1))
            Z2 = self.Theta_2.T @ Y1
            A2 = NeuralNetwork.linear(Z2)
            # cache[n] stores the intermediate values of the forward prop of the n-th piece of training data
            cache.append({'Y0': Y0, 'Z1': Z1, 'A1': A1,
                         'Y1': Y1, 'Z2': Z2, 'A2': A2})
            outputs[n] = A2

        return outputs, cache

    def backward_propagation(self, cache, Y):
        """Performs a backpropagation step of the NN

        Args:
            cache (dict): Cache of values from the forward prop step
            Y (np.array(float)): labels of inputs for `cache`, of size (N, output_unit)

        Returns:
            (np.array(float), np.array(float)): the gradients for `Theta_1`, `Theta_2`
        """
        # Y used to be shape (1, N)
        # I changed Y to shape (N, output_unit) so that Y[n] is a vector label of the n-th training data point
        N = Y.shape[1]

        # Delta_2[n][j] is the derivative of loss function J_n w.r.t. the linear combiner of the j-th neuron of the
        # final layer for the n-th piece of training data, 0-indexed
        Delta_2 = np.zeros(N, self.output_unit)

        for n in range(N):
            # Can iterating over j be avoided with numpy magic?
            for j in range(self.output_unit):
                # See (18.24) p. 890
                Delta_2[n][j] += (Y[n][j] - cache[n]('A2')[j]) * self.linearprime(cache[n]('Z2')[j])

        # Delta_1[n][j] is the derivative of loss function J_n w.r.t. the linear combiner of the j-th neuron of the
        # middle layer for the n-th piece of training data
        Delta_1 = np.zeros(N, self.hidden_unit)
        E = np.zeros(N, self.hidden_unit)
        for n in range(N):
            for j in range(self.hidden_unit):
                # See (18.31) p. 891
                # j+1 because first row of Theta_2 is for biases. Theta_2[j+1] corr. to the weights to the output layer
                # from the hidden layer's j-th neuron.
                E[n][j] += np.dot(Delta_2[n], self.Theta_2[j+1])
                # See (18.32) p. 891
                Delta_1[n][j] += E[n][j] * self.tanhprime(cache[n]('Z1')[j])

        # Grad_i has the gradients for Theta_i

        Grad_1 = np.zeros(self.input_unit + 1, self.hidden_unit)
        Grad_2 = np.zeros(self.hidden_unit + 1, self.output_unit)

        # Grad_2[i+1][j] is the gradient of the link between the i-th neuron of the hidden layer and the j-th neuron of
        # the output layer, while Grad_2[0][j] is the gradient of the bias of the j-th neuron of the output layer
        for j in range(self.output_unit):
            for i in range(self.hidden_unit + 1):
                for n in range(N):
                    # TODO sum this better
                    # See (18.21) p. 890
                    Grad_2[i][j] += Delta_2[n][j] * cache[n]('Y1')[i]

        # Grad_1[i+1][j] is the gradient of the link between the i-th neuron of the input layer and the j-th neuron of
        # the hidden layer, while Grad_1[0][j] is the gradient of the bias of the j-th neuron of the hidden layer
        for j in range(self.hidden_unit):
            for i in range(self.input_unit + 1):
                for n in range(N):
                    # See (18.21) p. 890
                    Grad_1[i][j] += Delta_1[n][j] * cache[n]('Y0')[i]

        return Grad_1, Grad_2

    def gradient_descent(self, grads, learning_rate=0.01):
        """Performs gradient descent, given a batch of gradients

        Args:
            grads (np.array(float), np.array(float)): gradients from backpropagation
            learning_rate (float, optional): learning rate for gradient descent. Defaults to 0.01.
        """
        self.Theta_1 = self.Theta_1 - learning_rate * grads[0]
        self.Theta_2 = self.Theta_2 - learning_rate * grads[1]

    # TODO implement a training session