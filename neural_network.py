

import random   # who's ready
import pickle   # for random
import math     # pickle
from datetime import datetime, timedelta
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
        self.f_weights = None
        self.b_weights = None
        self.act_func = None
        self.deriv_func = None
        self.activation = 0
        self.bias = 0
        self.delta = 0
        self.delta_sum = 0

    def compute_activation(self):
        weights = [w.value for w in self.b_weights]
        activations = [w.b_node.activation for w in self.b_weights]
        weighted_inputs = [w*a for w, a in zip(weights, activations)]
        combined_inputs = sum(weighted_inputs) + self.bias
        self.activation = self.act_func(combined_inputs)

    def compute_error_gradient(self):
        weights = [w.value for w in self.f_weights]
        deltas = [w.f_node.delta for w in self.f_weights]
        weighted_deltas = [w*d for w, d in zip(weights, deltas)]
        combined_deltas = sum(weighted_deltas)
        self.delta = self.deriv_func(self.activation) * combined_deltas

    def descend_bias_gradient(self, learning_rate):
        error_derivative = self.delta
        self.bias -= error_derivative * learning_rate


class Architect:

    def __init__(self, shape):
        self.nodes = self.build_node_structure(shape)
        self.weights = self.build_weight_structure(shape)
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
        self.samples = self.get_samples()
        self.start_time = time.time()
        self.current_time = time.time()
        self.ev = Evaluator(neural_network, self)

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
        adjusted_learning_rate = learning_rate / batch_size
        while self.current_time - self.start_time < training_time:
            self.current_time = time.time()
            if self.ev.report_is_due(): self.ev.basic_report()
            for sample in random.sample(self.samples, batch_size):
                self.nn.forward_pass(sample['pixels'])
                self.backpropagate(sample['one_hot'])
                self.update_delta_sums()
            self.descend_weight_gradients(adjusted_learning_rate)
            self.descend_bias_gradients(adjusted_learning_rate)
            self.reset_delta_sums()
        self.ev.final_report(learning_rate, batch_size)


class Evaluator:

    def __init__(self, neural_network, trainer):
        self.nn = neural_network
        self.tr = trainer
        self.last_report = 0
        self.report_freq = 10
        self.samples = self.get_samples()
        self.init_report = self.evaluate_network(1000)

    @staticmethod
    def get_samples():
        with open("mnist_data.pkl", "rb") as f:
            mnist_data = pickle.load(f)
        return mnist_data['testing_samples']

    def is_correct_prediction(self, targets):
        return targets == self.nn.one_hot

    def get_mean_squared_error(self, targets):
        squared_errors = [(p-t)**2 for p, t in zip(self.nn.output, targets)]
        return sum(squared_errors) / len(targets)

    def evaluate_network(self, num_of_samples=300):
        acc_sum = 0
        mse_sum = 0
        for _ in range(num_of_samples):
            sample = random.choice(self.samples)
            self.nn.set_predictions(sample['pixels'])
            acc_sum += self.is_correct_prediction(sample['one_hot'])
            mse_sum += self.get_mean_squared_error(sample['one_hot'])
        avg_acc = acc_sum / num_of_samples
        avg_mse = mse_sum / num_of_samples
        return avg_acc, avg_mse

    def report_is_due(self):
        if self.tr.current_time - self.last_report > self.report_freq:
            self.last_report = self.tr.current_time
            return True

    @staticmethod
    def time_of_day(unix_time):
        return datetime.fromtimestamp(unix_time).strftime("%H:%M:%S")

    @staticmethod
    def elapsed(seconds):
        return str(timedelta(seconds=round(seconds)))

    def basic_report(self):
        acc, mse = self.evaluate_network()
        elapsed_time = self.elapsed(self.tr.current_time - self.tr.start_time)
        print(f'Acc: {round(acc, 3)}, '
              f'MSE: {round(mse, 3)}, '
              f'Elapsed Time: {elapsed_time}')

    def final_report(self, learning_rate, batch_size):
        init_acc, init_mse = self.init_report
        final_acc, final_mse = self.evaluate_network(1000)
        print()
        print('#####################  -  Final Report -  #####################')
        print(f'Start Time: {self.time_of_day(self.tr.start_time)},',
              f'End Time: {self.time_of_day(time.time())},',
              f'Elapsed Time: {self.elapsed(time.time() - self.tr.start_time)}')
        print(f'Learning Rate: {round(learning_rate, 3)},',
              f'Batch Size: {batch_size},',
              f'Adj Learning Rate: {round((learning_rate / batch_size), 5)}')
        print(f'Starting Accuracy: {round(init_acc, 3)},',
              f'Starting Mean Squared Error: {round(init_mse, 3)}')
        print(f'Final Accuracy:    {round(final_acc, 3)},',
              f'Final Mean Squared Error:    {round(final_mse, 3)}')
        print('###############################################################')


if __name__ == '__main__':

    nn = NeuralNetwork([784, 16, 16, 10])
    nn.train_network(0.1, 32, 10)


'''
I need to be able to save my weights.
Normalize for batch size.
Change my one hot encoded data to floats.
Remove node references in nodes.
'''

