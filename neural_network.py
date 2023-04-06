

import random   # who's ready
import pickle   # for random
import math     # pickle
import time     # math time?


class Weight:

    def __init__(self):
        self.value = 0
        self.b_node = None
        self.f_node = None

    def descend_weight_gradient(self, learning_rate):
        error_derivative = self.f_node.delta * self.b_node.activation
        self.value -= error_derivative * learning_rate


class Node:

    def __init__(self):
        self.f_nodes = None
        self.b_nodes = None
        self.f_weights = None
        self.b_weights = None
        self.act_func = None
        self.deriv_func = None
        self.activation = 0
        self.bias = 0
        self.delta = 0
        self.delta_sum = 0

    def compute_activation(self):
        b_activations = [b_node.activation for b_node in self.b_nodes]
        b_weights = [b_weight.value for b_weight in self.b_weights]
        weighted_inputs = [a*w for a, w in zip(b_activations, b_weights)]
        combined_inputs = sum(weighted_inputs) + self.bias
        self.activation = self.act_func(combined_inputs)

    def compute_error_gradient(self):
        f_deltas = [f_node.delta for f_node in self.f_nodes]
        f_weights = [f_weight.value for f_weight in self.f_weights]
        weighted_deltas = [d*w for d, w in zip(f_deltas, f_weights)]
        combined_deltas = sum(weighted_deltas)
        self.delta = self.deriv_func(self.activation) * combined_deltas

    def descend_bias_gradient(self, learning_rate):
        error_derivative = self.delta
        self.bias -= error_derivative * learning_rate


class Architect:

    def __init__(self, shape):
        self.nodes = self.build_node_structure(shape)
        self.weights = self.build_weight_structure(shape)
        self.set_node_references_in_nodes(self.nodes)
        self.set_weight_references_in_nodes(self.nodes, self.weights)
        self.set_node_references_in_weights(self.nodes, self.weights)
        self.initialize_weight_values(self.weights)
        self.set_act_and_deriv_funcs(self.nodes)

    @staticmethod
    def build_node_structure(shape):
        node_layers = []
        for size in shape:
            node_layers.append([Node() for _ in range(size)])
        return node_layers

    @staticmethod
    def build_weight_structure(shape):
        weight_layers = []
        for back, fore in zip(shape[:-1], shape[1:]):
            node_weights = []
            for _ in range(back):
                node_weights.append([Weight() for z in range(fore)])
            weight_layers.append(node_weights)
        return weight_layers

    @staticmethod
    def set_node_references_in_nodes(n_layers):
        for b_n_layer, f_n_layer in zip(n_layers[1:-1], n_layers[2:]):
            for b_node in b_n_layer:
                b_node.f_nodes = f_n_layer
        for b_n_layer, f_n_layer in zip(n_layers[0:-1], n_layers[1:]):
            for f_node in f_n_layer:
                f_node.b_nodes = b_n_layer

    @staticmethod
    def set_weight_references_in_nodes(n_layers, w_layers):
        for n_layer, w_layer in zip(n_layers[1:-1], w_layers[1:]):
            for node, weights in zip(n_layer, w_layer):
                node.f_weights = weights
        for n_layer, w_layer in zip(n_layers[1:], w_layers):
            for i, node in enumerate(n_layer):
                node.b_weights = [w[i] for w in w_layer]

    @staticmethod
    def set_node_references_in_weights(n_layers, w_layers):
        zipped_layers = zip(n_layers[:-1], w_layers, n_layers[1:])
        for b_n_layer, w_layer, f_n_layer in zipped_layers:
            for b_node, weights in zip(b_n_layer, w_layer):
                for weight, f_node in zip(weights, f_n_layer):
                    weight.b_node = b_node
                    weight.f_node = f_node

    @staticmethod
    def initialize_weight_values(w_layers):
        def he_wt_init(n): return random.gauss(0, math.sqrt(2 / n))
        for w_layer in w_layers:
            for weights in w_layer:
                for weight in weights:
                    weight.value = he_wt_init(len(w_layer))

    @staticmethod
    def set_act_and_deriv_funcs(n_layers):
        def relu_act(x): return max(0, x)
        def relu_deriv(x): return 1 if x > 0 else 0
        def identity(x): return x
        for n_layer in n_layers[1:-1]:
            for node in n_layer:
                node.act_func = relu_act
                node.deriv_func = relu_deriv
        for node in n_layers[-1]:
            node.act_func = identity


class NeuralNetwork:

    def __init__(self, shape):
        structure = Architect(shape)
        self.weights = structure.weights
        self.nodes = structure.nodes
        self.input = []
        self.output = []
        self.prediction = 0
        self.one_hot = []
        self.train_network = Trainer(self).train_network

    def set_features(self, features):
        self.input = features
        for node, feature in zip(self.nodes[0], features):
            node.activation = feature

    def forward_pass(self, features):
        self.set_features(features)
        for layer in self.nodes[1:]:
            for node in layer:
                node.compute_activation()

    def set_predictions(self, features):
        if not self.input == features:
            self.forward_pass(features)
            self.output = [n.activation for n in self.nodes[-1]]
            self.prediction = self.output.index(max(self.output))
            self.one_hot = [float(i == self.prediction) for i in range(10)]

    def predict(self, features):
        self.set_predictions(features)
        return self.prediction


class Trainer:

    def __init__(self, neural_network):
        self.nn = neural_network
        self.eval = Evaluator(neural_network)
        self.eval_freq = 10
        self.samples = self.get_samples()

    @staticmethod
    def get_samples():
        with open("mnist_data.pkl", "rb") as f:
            mnist_data = pickle.load(f)
        return mnist_data["training_samples"]

    def set_output_deltas(self, targets):
        for node, target in zip(self.nn.nodes[-1], targets):
            node.delta = node.activation - target

    def backpropagate(self, targets):
        self.set_output_deltas(targets)
        for layer in reversed(self.nn.nodes[1:-1]):
            for node in layer:
                node.compute_error_gradient()

    def update_delta_sums(self):
        for layer in self.nn.nodes[1:]:
            for node in layer:
                node.delta_sum += node.delta

    def descend_weight_gradients(self, learning_rate):
        for layer in self.nn.weights:
            for n_weights in layer:
                for weight in n_weights:
                    weight.descend_weight_gradient(learning_rate)

    def descend_bias_gradients(self, learning_rate):
        for layer in self.nn.nodes[1:]:
            for node in layer:
                node.descend_bias_gradient(learning_rate)

    def reset_delta_sums(self):
        for layer in self.nn.nodes[1:]:
            for node in layer:
                node.delta_sum = 0

    def train_network(self, learning_rate, batch_size, training_time):
        start_time = time.time()
        learning_rate /= batch_size
        while time.time() < start_time + training_time:
            self.eval.report_progress_if_interval(self.eval_freq)
            for sample in random.sample(self.samples, batch_size):
                self.nn.forward_pass(sample['pixels'])
                self.backpropagate(sample['one_hot'])
                self.update_delta_sums()
            self.descend_weight_gradients(learning_rate)
            self.descend_bias_gradients(learning_rate)
            self.reset_delta_sums()


class Evaluator:

    def __init__(self, neural_network):
        self.nn = neural_network
        self.samples = self.get_samples()
        self.last_report = 0

    @staticmethod
    def get_samples():
        with open("mnist_data.pkl", "rb") as f:
            mnist_data = pickle.load(f)
        return mnist_data['testing_samples']

    def get_accuracy(self, targets):
        return targets == self.nn.one_hot

    def get_ms_error(self, targets):
        squared_errors = [(p-t)**2 for p, t in zip(self.nn.output, targets)]
        return sum(squared_errors) / len(targets)

    def test_network(self, num_of_samples=300):
        acc_sum = 0
        mse_sum = 0
        for _ in range(num_of_samples):
            sample = random.choice(self.samples)
            self.nn.set_predictions(sample['pixels'])
            acc_sum += self.get_accuracy(sample['one_hot'])
            mse_sum += self.get_ms_error(sample['one_hot'])
        avg_acc = acc_sum / num_of_samples
        avg_mse = mse_sum / num_of_samples
        return avg_acc, avg_mse

    def report_progress_if_interval(self, eval_freq):
        if time.time() - self.last_report >= eval_freq:
            print(self.test_network())
            self.last_report = time.time()


if __name__ == '__main__':

    nn = NeuralNetwork([784, 16, 16, 10])
    nn.train_network(0.1, 32, 60)


'''
I need to be able to save my weights.
Normalize for batch size.
Change my one hot encoded data to floats.
Remove node references in nodes.
'''

