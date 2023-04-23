import os
import math
import time
import random
import pickle
import gzip
from datetime import datetime, timedelta


class Neuron:

    def __init__(self):
        self.inc_conns = None
        self.out_conns = None
        self.activ_func = None
        self.deriv_func = None
        self.activation = 0
        self.bias = 0
        self.error_gradient = 0
        self.batch_gradient = 0

    def compute_activation(self):
        inc_weights = [ic.weight for ic in self.inc_conns]
        inc_activs = [ic.source_node.activation for ic in self.inc_conns]
        weighted_sum = sum([w*a for w, a in zip(inc_weights, inc_activs)])
        self.activation = self.activ_func(weighted_sum + self.bias)

    def compute_error_gradient(self):
        out_weights = [oc.weight for oc in self.out_conns]
        out_err_grd = [oc.target_node.error_gradient for oc in self.out_conns]
        weighted_sum = sum([w*d for w, d in zip(out_weights, out_err_grd)])
        self.error_gradient = self.deriv_func(self.activation) * weighted_sum

    def bias_gradient_descent(self, learning_rate):
        self.bias -= self.batch_gradient * learning_rate


class Connection:

    def __init__(self):
        self.weight = 0
        self.source_node = None
        self.target_node = None

    def weight_gradient_descent(self, learning_rate):
        delta = self.source_node.activation * self.target_node.batch_gradient
        self.weight -= delta * learning_rate


class Architect:

    def __init__(self, shape):
        self.neurons = self.create_nodes(shape)
        self.connections = self.create_connections(shape)
        self.set_connection_references_in_nodes(self.neurons, self.connections)
        self.set_node_references_in_connections(self.neurons, self.connections)
        self.set_activation_and_derivative_functions_in_nodes(self.neurons)
        self.initialize_weight_values_in_connections(self.connections)

    @staticmethod
    def create_nodes(shape):
        node_layers = []
        for size in shape:
            node_layers.append([Neuron() for _ in range(size)])
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
                node.out_conns = connection_group
        for node_layer, conn_layer in zip(node_layers[1:], conn_layers):
            for i, node in enumerate(node_layer):
                node.inc_conns = [c[i] for c in conn_layer]

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
        def relu_activation(x):
            return max(0, x)
        def relu_derivative(x):
            return 1 if x > 0 else 0
        def identity(x):
            return x
        for node_layer in node_layers[1:-1]:
            for node in node_layer:
                node.activ_func = relu_activation
                node.deriv_func = relu_derivative
        for node in node_layers[-1]:
            node.activ_func = identity

    @staticmethod
    def initialize_weight_values_in_connections(conn_layers):
        def he_weight_init(n):
            return random.gauss(0, math.sqrt(2 / n))
        for conn_layer in conn_layers:
            nodes_in_prev_layer = len(conn_layer)
            for conn_group in conn_layer:
                for connection in conn_group:
                    connection.weight = he_weight_init(nodes_in_prev_layer)


class NeuralNetwork:

    def __init__(self, shape=None, model_name=None):
        architecture = self.create_arhitecture(shape, model_name)
        self.neurons = architecture.neurons
        self.connections = architecture.connections
        self.output = []
        self.one_hot = []
        self.prediction = 0
        self.train_network = Trainer(self).train_network

    @staticmethod
    def create_arhitecture(shape, model_name):
        if shape:
            return Architect(shape)
        else:
            return ModelIO().load_model(model_name)

    def update_features(self, features):
        for neuron, feature in zip(self.neurons[0], features):
            neuron.activation = feature

    def forward_pass(self, features):
        self.update_features(features)
        for layer in self.neurons[1:]:
            for neuron in layer:
                neuron.compute_activation()

    def update_predictions(self, features):
        self.forward_pass(features)
        self.output = [n.activation for n in self.neurons[-1]]
        self.prediction = self.output.index(max(self.output))
        self.one_hot = [float(i == self.prediction) for i in range(10)]

    def get_prediction(self, features):
        self.update_predictions(features)
        return self.prediction


class Trainer:

    def __init__(self, neural_network):
        self.nn = neural_network
        self.samples = self.get_samples()
        self.ev = Evaluator(neural_network)

    @staticmethod
    def get_samples():
        with gzip.open('mnist_data.pkl.gz', 'rb') as f:
            mnist_data = pickle.load(f)
        return mnist_data["training_samples"]

    def compute_and_set_output_deltas(self, targets):
        for node, target in zip(self.nn.neurons[-1], targets):
            node.error_gradient = node.activation - target

    def backpropagate(self):
        for layer in reversed(self.nn.neurons[1:-1]):
            for node in layer:
                node.compute_error_gradient()

    def accumulate_error_gradient_in_batch_gradient(self):
        for layer in self.nn.neurons[1:]:
            for node in layer:
                node.batch_gradient += node.error_gradient

    def adjust_biases_by_gradients(self, learning_rate):
        for node_layer in self.nn.neurons[1:]:
            for node in node_layer:
                node.bias_gradient_descent(learning_rate)

    def adjust_weights_by_gradients(self, learning_rate):
        for conn_layer in self.nn.connections:
            for conn_group in conn_layer:
                for connection in conn_group:
                    connection.weight_gradient_descent(learning_rate)

    def reset_delta_accumulator(self):
        for layer in self.nn.neurons[1:]:
            for node in layer:
                node.batch_gradient = 0

    def train_network(self, learning_rate, batch_size, train_mins, name=None):
        start_time = time.time()
        adjusted_learning_rate = learning_rate / batch_size
        try:
            while (time.time() - start_time) / 60 < train_mins:
                self.ev.if_report_frequency_print_basic_report(start_time)
                for sample in random.sample(self.samples, batch_size):
                    self.nn.forward_pass(sample['pixels'])
                    self.compute_and_set_output_deltas(sample['one_hot'])
                    self.backpropagate()
                    self.accumulate_error_gradient_in_batch_gradient()
                self.adjust_biases_by_gradients(adjusted_learning_rate)
                self.adjust_weights_by_gradients(adjusted_learning_rate)
                self.reset_delta_accumulator()
        except KeyboardInterrupt:
            print('Training stopped early by user.')
        finally:
            if name: ModelIO().save_model(self.nn, name)
            self.ev.print_final_report(learning_rate, batch_size, start_time)


class Evaluator:

    def __init__(self, neural_network):
        self.nn = neural_network
        self.report_freq = 10
        self.last_report = 0
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
        squared_errors = [(p - t) ** 2 for p, t in zip(self.nn.output, targets)]
        return sum(squared_errors) / len(targets)

    def evaluate_network(self, num_of_samples=200):
        acc_sum = 0
        mse_sum = 0
        for _ in range(num_of_samples):
            sample = random.choice(self.samples)
            self.nn.update_predictions(sample['pixels'])
            acc_sum += self.is_accurate_prediction(sample['one_hot'])
            mse_sum += self.get_mean_squared_error(sample['one_hot'])
        avg_acc = acc_sum / num_of_samples
        avg_mse = mse_sum / num_of_samples
        return avg_acc, avg_mse

    @staticmethod
    def format_time(unix_time):
        return datetime.fromtimestamp(unix_time).strftime("%H:%M:%S")

    @staticmethod
    def elapsed(start, end):
        return str(timedelta(seconds=round(end - start)))

    def if_report_frequency_print_basic_report(self, start_time):
        if time.time() - self.last_report > self.report_freq:
            self.last_report = time.time()
            acc, mse = self.evaluate_network()
            elapsed = self.elapsed(start_time, time.time())
            print(f'Acc: {round(acc, 3)}, '
                  f'MSE: {round(mse, 3)}, '
                  f'Elapsed Time: {elapsed}')

    def print_final_report(self, learning_rate, batch_size, start_time):
        init_acc, init_mse = self.init_report
        final_acc, final_mse = self.evaluate_network(2000)
        elapsed = self.elapsed(start_time, time.time())
        print('#####################  -  Final Report -  #####################')
        print(f'Start Time: {self.format_time(start_time)},',
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


class ModelIO:

    @staticmethod
    def get_weights(connections):
        return [[[c.weight for c in g] for g in l] for l in connections]

    @staticmethod
    def get_biases(nodes):
        return [[n.bias for n in l] for l in nodes]

    @staticmethod
    def set_weights(target, source):
        for t_layer, s_layer in zip(target, source):
            for t_group, s_group in zip(t_layer, s_layer):
                for t_conn, s_weight in zip(t_group, s_group):
                    t_conn.weight = s_weight

    @staticmethod
    def set_biases(target, source):
        for t_layer, s_layer in zip(target, source):
            for t_node, s_bias in zip(t_layer, s_layer):
                t_node.bias = s_bias

    def save_model(self, nn, model):
        models = {}
        if os.path.isfile('models.pkl'):
            with open('models.pkl', 'rb') as f:
                models = pickle.load(f)
        biases = self.get_biases(nn.nodes)
        weights = self.get_weights(nn.connections)
        models[model] = (biases, weights)
        with open('models.pkl', 'wb') as f:
            pickle.dump(models, f)

    def load_model(self, model):
        with open('models.pkl', 'rb') as f:
            models = pickle.load(f)
        shape = [len(layer) for layer in models[model][1]]
        nn = Architect(shape)
        self.set_weights(nn.connections, models[model][0])
        self.set_biases(nn.neurons, models[model][1])
        return nn


if __name__ == '__main__':
    nn = NeuralNetwork([784, 64, 32, 10])
    nn.train_network(0.032, 3, 1)
