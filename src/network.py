import numpy as np

class Network:

    def __init__(self, n_inputs=4, n_hidden=[5, 4], n_outputs=3):
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs

        if (n_hidden == []):
            layers = [self.n_inputs] + [self.n_outputs]
        else:
            layers = [self.n_inputs] + self.n_hidden + [self.n_outputs]

        self.layers = layers
        self.weights = []

    def initialize_weights(self):
        for i in range(len(self.layers)-1):
            self.weights.append(np.random.rand(self.layers[i], self.layers[i+1]))

    def feedforward(self, inputs):
        activation = inputs

        for i in self.weights:
            net_inputs = np.dot(activation, i)
            activation = self.g(net_inputs)

        return activation

    def g(self, x):
        return 1 / (1 + np.exp(-x))


# if __name__ == '__main__':
#     net = Network(n_hidden=[])
#     net.initialize_weights()
    
#     inputs = np.random.rand(net.n_inputs)

#     outputs = net.feedforward(inputs)
#     print("Network input is {}".format(inputs))
#     print("Network output is {}".format(outputs))