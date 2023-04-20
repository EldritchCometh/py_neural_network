import os
import math
import time
import random
import pickle
import gzip
from datetime import datetime, timedelta


class Node:

    def __init__(self):
        self.incoming_connections = None
        self.outgoing_connections = None
        self.activation_function = None
        self.derivative_function = None
        self.activation = 0
        self.bias = 0
        self.delta = 0
        self.delta_accumulator = 0

    def compute_activation(self):
        weights = [ic.weight for ic in self.incoming_connections]
        activs = [ic.source_node.activation for ic in self.incoming_connections]
        weighted_sum = sum([w * a for w, a in zip(weights, activs)])
        self.activation = self.activation_function(weighted_sum + self.bias)

    def compute_error_gradient(self):
        weights = [oc.weight for oc in self.outgoing_connections]
        deltas = [oc.target_node.delta for oc in self.outgoing_connections]
        weighted_sum = sum([w * d for w, d in zip(weights, deltas)])
        self.delta = self.derivative_function(self.activation) * weighted_sum

    def descend_bias_gradient(self, learning_rate):
        self.bias -= self.delta_accumulator * learning_rate


class Connection:

    def __init__(self):
        self.weight = 0
        self.source_node = None
        self.target_node = None

    def descend_weight_gradient(self, learning_rate):
        delta = self.source_node.activation * self.target_node.delta_accumulator
        self.weight -= delta * learning_rate


class Architect:

    def __init__(self, shape):
        self.nodes = self.create_nodes(shape)
        self.connections = self.create_connections(shape)
        self.set_connection_references_in_nodes(self.nodes, self.connections)
        self.set_node_references_in_connections(self.nodes, self.connections)
        self.set_activation_and_derivative_functions_in_nodes(self.nodes)
        self.initialize_weight_values_in_connections(self.connections)

    @staticmethod
    def create_nodes(shape):
        node_layers = []
        for size in shape:
            node_layers.append([Node() for _ in range(size)])
        return node_layers

    @staticmethod
    def create_connections(shape):
        connection_layers = []
        for back, fore in zip(shape[:-1], shape[1:]):
            node_group = []
            for _ in range(back):
                node_group.append([Connection() for _ in range(fore)])
            connection_layers.append(node_group)
        return connection_layers

    @staticmethod
    def set_connection_references_in_nodes(node_layers, conn_layers):
        for node_layer, conn_layer in zip(node_layers[1:-1], conn_layers[1:]):
            for node, connection_group in zip(node_layer, conn_layer):
                node.outgoing_connections = connection_group
        for node_layer, conn_layer in zip(node_layers[1:], conn_layers):
            for i, node in enumerate(node_layer):
                node.incoming_connections = [c[i] for c in conn_layer]

    @staticmethod
    def set_node_references_in_connections(node_layers, conn_layers):
        zipped_layers = zip(node_layers[:-1], conn_layers, node_layers[1:])
        for b_node_layer, conn_layer, f_node_layer in zipped_layers:
            for b_node, connection_group in zip(b_node_layer, conn_layer):
                for connection, f_node in zip(connection_group, f_node_layer):
                    connection.source_node = b_node
                    connection.target_node = f_node

    @staticmethod
    def set_activation_and_derivative_functions_in_nodes(node_layers):
        def relu_activation(x): return max(0, x)
        def relu_derivative(x): return 1 if x > 0 else 0
        def identity(x): return x
        for node_layer in node_layers[1:-1]:
            for node in node_layer:
                node.activation_function = relu_activation
                node.derivative_function = relu_derivative
        for node in node_layers[-1]:
            node.activation_function = identity

    @staticmethod
    def initialize_weight_values_in_connections(conn_layers):
        def he_weight_init(n): return random.gauss(0, math.sqrt(2 / n))
        for conn_layer in conn_layers:
            nodes_in_prev_layer = len(conn_layer)
            for conn_group in conn_layer:
                for connection in conn_group:
                    connection.weight = he_weight_init(nodes_in_prev_layer)


class NeuralNetwork:

    def __init__(self, shape=None, name=None):
        if shape: model = Architect(shape)
        else: model = self.load_model(name)
        self.weights = model.connections
        self.nodes = model.nodes
        self.input = []
        self.output = []
        self.prediction = 0
        self.one_hot = []
        self.train_network = Trainer(self).train_network

    @staticmethod
    def load_model(name):
        with open('models.pkl', 'rb') as f:
            models = pickle.load(f)
            return models[name]

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
        self.start_time = time.time()
        self.current_time = time.time()
        self.samples = self.get_samples()
        self.ev = Evaluator(neural_network, self)

    @staticmethod
    def get_samples():
        with gzip.open('mnist_data.pkl.gz', 'rb') as f:
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

    def update_delta_accumulator(self):
        for layer in self.nn.nodes[1:]:
            for node in layer:
                node.delta_accumulator += node.delta

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
                node.delta_accumulator = 0

    def save_model(self, name):
        class MiniArchitect:
            weights = None
            biases = None
        with open('models.pkl', 'wb') as f:
            with open('models.pkl', 'rb') as g:
                models = pickle.load(g)
                mini_model = MiniArchitect()
                mini_model.weights = self.nn.weights
                mini_model.nodes = self.nn.nodes
                models[name] = mini_model
            pickle.dump(models, f)

    def train_network(self, learning_rate, batch_size, train_time, name=None):
        adjusted_learning_rate = learning_rate / batch_size
        while self.current_time - self.start_time < train_time:
            self.current_time = time.time()
            self.ev.if_report_frequency_print_basic_report()
            for sample in random.sample(self.samples, batch_size):
                self.nn.forward_pass(sample['pixels'])
                self.backpropagate(sample['one_hot'])
                self.update_delta_accumulator()
            self.descend_weight_gradients(adjusted_learning_rate)
            self.descend_bias_gradients(adjusted_learning_rate)
            self.reset_delta_sums()
        self.ev.print_final_report(learning_rate, batch_size)
        if name: self.save_model(name)


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
        with gzip.open('mnist_data.pkl.gz', 'rb') as f:
            mnist_data = pickle.load(f)
        return mnist_data['testing_samples']

    def is_accurate_prediction(self, targets):
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
            acc_sum += self.is_accurate_prediction(sample['one_hot'])
            mse_sum += self.get_mean_squared_error(sample['one_hot'])
        avg_acc = acc_sum / num_of_samples
        avg_mse = mse_sum / num_of_samples
        return avg_acc, avg_mse

    @staticmethod
    def format_time(unix_time):
        return datetime.fromtimestamp(unix_time).strftime("%H:%M:%S")

    @staticmethod
    def format_elapsed(seconds):
        return str(timedelta(seconds=round(seconds)))

    def if_report_frequency_print_basic_report(self):
        if self.tr.current_time - self.last_report > self.report_freq:
            self.last_report = self.tr.current_time
            acc, mse = self.evaluate_network()
            elapsed = self.tr.current_time - self.tr.start_time
            formatted = self.format_elapsed(elapsed)
            print(f'Acc: {round(acc, 3)}, '
                  f'MSE: {round(mse, 3)}, '
                  f'Elapsed Time: {formatted}')

    def print_final_report(self, learning_rate, batch_size):
        init_acc, init_mse = self.init_report
        final_acc, final_mse = self.evaluate_network(1000)
        elapsed = self.format_elapsed(self.tr.current_time - self.tr.start_time)
        print('#####################  -  Final Report -  #####################')
        print(f'Start Time: {self.format_time(self.tr.start_time)},',
              f'End Time: {self.format_time(time.time())},',
              f'Elapsed Time: {elapsed}')
        print(f'Learning Rate: {round(learning_rate, 3)},',
              f'Batch Size: {batch_size},',
              f'Adj Learning Rate: {round((learning_rate / batch_size), 5)}')
        print(f'Pre-training Accuracy:  {round(init_acc, 3)},',
              f'Pre-training MSE:  {round(init_mse, 3)}')
        print(f'Post-training Accuracy: {round(final_acc, 3)},',
              f'Post-training MSE: {round(final_mse, 3)}')
        print('###############################################################')


if __name__ == '__main__':

    # need to add back all the features obviously
    # but also need to fix places where it uses delta to accumulated delta

    nn = NeuralNetwork([784, 64, 32, 10])
    nn.train_network(0.032, 3, 60)