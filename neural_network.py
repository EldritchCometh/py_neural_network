

import math
import random
import pickle
import time


class Weight:

    def __init__(self):
        self.value = 0
        self.b_node = None
        self.f_node = None

    def descend_weight_gradient(self, learning_rate):
        self.value -= self.b_node.activation * self.f_node.delta * learning_rate


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
        w_sum = sum([a*w for a, w in zip(b_activations, b_weights)])
        self.activation = self.act_func(w_sum + self.bias)

    def compute_error_gradient(self):
        f_deltas = [f_node.delta for f_node in self.f_nodes]
        f_weights = [f_weight.value for f_weight in self.f_weights]
        w_sum = sum([d*w for d, w in zip(f_deltas, f_weights)])
        self.delta = self.deriv_func(self.activation) * w_sum

    def descend_bias_gradient(self, learning_rate):
        self.bias -= self.delta * learning_rate


class Architect:

    def __init__(self, shape, functions):
        self.nodes = self.build_node_layers(shape)
        self.weights = self.build_weight_layers(shape)
        self.set_node_references_in_nodes(self.nodes)
        self.set_weight_references_in_nodes(self.nodes, self.weights)
        self.set_node_references_in_weights(self.nodes, self.weights)
        self.initialize_weight_values(self.weights, functions)
        self.set_act_and_deriv_funcs(self.nodes, functions)

    @staticmethod
    def build_node_layers(shape):
        node_layers = []
        for size in shape:
            node_layers.append([Node() for _ in range(size)])
        return node_layers

    @staticmethod
    def build_weight_layers(shape):
        w_layers = []
        for back, fore in zip(shape[:-1], shape[1:]):
            node_weights = []
            for _ in range(back):
                node_weights.append([Weight() for z in range(fore)])
            w_layers.append(node_weights)
        return w_layers

    @staticmethod
    def set_node_references_in_nodes(n_layers):
        for back, fore in zip(n_layers[1:-1], n_layers[2:]):
            for b_node in back:
                b_node.f_nodes = fore
        for back, fore in zip(n_layers[0:-1], n_layers[1:]):
            for f_node in fore:
                f_node.b_nodes = back

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
        for b_node_layer, w_layer, f_node_layer in zipped_layers:
            for b_node, node_weights in zip(b_node_layer, w_layer):
                for weight, f_node in zip(node_weights, f_node_layer):
                    weight.b_node = b_node
                    weight.f_node = f_node

    @staticmethod
    def initialize_weight_values(w_layers, functions):
        for w_layer in w_layers:
            for n_weights in w_layer:
                for weight in n_weights:
                    weight.value = functions['weight_init'](len(w_layer))

    @staticmethod
    def set_act_and_deriv_funcs(n_layers, functions):
        for layer in n_layers[1:-1]:
            for node in layer:
                node.act_func = functions['hidden_act']
                node.deriv_func = functions['hidden_deriv']
        for node in n_layers[-1]:
            node.act_func = functions['output_act']


class Trainer:

    def __init__(self, neural_network):
        self.test_network = neural_network.test_network
        self.nodes = neural_network.nodes
        self.weights = neural_network.weights
        self.forward_pass = neural_network.forward_pass
        self.samples = self.get_samples()

    @staticmethod
    def get_samples():
        with open("mnist_data.pkl", "rb") as f:
            mnist_data = pickle.load(f)
        return mnist_data["training_samples"]

    def set_output_deltas(self, targets):
        for node, target in zip(self.nodes[-1], targets):
            node.delta = node.activation - target

    def backpropagate(self, targets):
        self.set_output_deltas(targets)
        for layer in reversed(self.nodes[1:-1]):
            for node in layer:
                node.compute_error_gradient()

    def update_delta_sums(self):
        for layer in self.nodes[1:]:
            for node in layer:
                node.delta_sum += node.delta

    def descend_weight_gradients(self, l_rate):
        for layer in self.weights:
            for node_weights in layer:
                for weight in node_weights:
                    weight.descend_weight_gradient(l_rate)

    def descend_bias_gradients(self, l_rate):
        for layer in self.nodes[1:]:
            for node in layer:
                node.descend_bias_gradient(l_rate)

    def reset_delta_sums(self):
        for layer in self.nodes[1:]:
            for node in layer:
                node.delta_sum = 0

    def train_network(self, batch_size, l_rate, train_time, eval_freq):
        start_time, last_report = time.time(), time.time()
        while time.time() < start_time + train_time:
            for sample in random.sample(self.samples, batch_size):
                self.forward_pass(sample['pixels'])
                self.backpropagate(sample['one_hot'])
                self.update_delta_sums()
            self.descend_weight_gradients(l_rate)
            self.descend_bias_gradients(l_rate)
            self.reset_delta_sums()
            if time.time() - last_report > eval_freq:
                print(self.test_network())
                last_report = time.time()


class Evaluator:

    def __init__(self, neural_network):
        self.nodes = neural_network.nodes
        self.forward_pass = neural_network.forward_pass
        self.predict = neural_network.predict
        self.samples = self.get_samples()

    @staticmethod
    def get_samples():
        with open("mnist_data.pkl", "rb") as f:
            mnist_data = pickle.load(f)
        return mnist_data["testing_samples"]

    def get_accuracy(self, features, targets):
        return self.predict(features) == targets

    def get_ms_error(self, features, targets):
        prediction = [float(i == self.predict(features)) for i in range(10)]
        squared_errors = [(p - t) ** 2 for p, t in zip(prediction, targets)]
        return sum(squared_errors) / len(targets)

    def test_network(self, num_of_samples=100):
        acc_sum = 0
        mse_sum = 0
        for _ in range(num_of_samples):
            sample = random.choice(self.samples)
            acc_sum += self.get_accuracy(sample['pixels'], sample['label'])
            mse_sum += self.get_ms_error(sample['pixels'], sample['one_hot'])
        avg_acc = acc_sum / num_of_samples
        avg_mse = mse_sum / num_of_samples
        return avg_acc, avg_mse


class NeuralNetwork:

    def __init__(self, shape, functions):
        structure = Architect(shape, functions)
        self.weights = structure.weights
        self.nodes = structure.nodes
        self.test_network = Evaluator(self).test_network
        self.train_network = Trainer(self).train_network
        self.input = []
        self.output = []

    def set_features(self, features):
        self.input = features
        for node, feature in zip(self.nodes[0], features):
            node.activation = feature

    def forward_pass(self, features):
        if self.input == features:
            return self.output
        self.set_features(features)
        for layer in self.nodes[1:]:
            for node in layer:
                node.compute_activation()
        self.output = [n.activation for n in self.nodes[-1]]

    def predict(self, features):
        self.forward_pass(features)
        return self.output.index(max(self.output))


if __name__ == '__main__':
    functions = {
        'hidden_act': (lambda x: max(0, x)),
        'hidden_deriv': (lambda x: 1 if x > 0 else 0),
        'output_act': (lambda x: x),
        'weight_init': (lambda n: random.gauss(0, math.sqrt(2 / n)))}
    nn = NeuralNetwork([784, 16, 16, 10], functions)
    nn.train_network(32, 0.01, 120, 5)

    '''
    reporating frequency should be a paramater and should be time based
    I want easier access to all of my hyperparamaters rather than searching
    around in the code
    I need to be able to save my weights.
    I would like a way to train for a specified amount of time rather than 
    for iterations
    Eliminate delta and error duplication in output node
    Normalize for batch size.
    '''
    '''
    def get_training_prediction(self):
        return [n.activation for n in self.n_layers[-1]]

    def get_one_hot_prediction(self):
        i_pred = self.get_int_prediction()
        return [1 if i == i_pred else 0 for i in range(10)]
    '''
    '''
    nn[1][0].b_weights = [-34.4]
    nn[1][0].bias = 2.14
    nn[1][1].b_weights = [-2.52]
    nn[1][1].bias = 1.29
    nn[2][0].b_weights = [-1.3, 2.28]
    nn[2][0].bias = -0.58
    nn.forward([0.5])
    print(nn.get_training_prediction())
    print(nn.get_human_prediction())
    '''