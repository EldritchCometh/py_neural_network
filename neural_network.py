import os
import sys
import gzip
import math
import pickle
import random
from time import time
from datetime import datetime, timedelta


class Model:

    @staticmethod
    def get_biases(neurons):
        return [[n.bias for n in l] for l in neurons]

    @staticmethod
    def get_weights(connections):
        return [[[c.weight for c in g] for g in l] for l in connections]

    @staticmethod
    def set_biases(target, source):
        for t_layer, s_layer in zip(target, source):
            for t_neuron, s_bias in zip(t_layer, s_layer):
                t_neuron.bias = s_bias

    @staticmethod
    def set_weights(target, source):
        for t_layer, s_layer in zip(target, source):
            for t_group, s_group in zip(t_layer, s_layer):
                for t_conn, s_weight in zip(t_group, s_group):
                    t_conn.weight = s_weight

    @classmethod
    def save_model(cls, nn):
        models = {}
        if os.path.isfile('models.pkl'):
            with open('models.pkl', 'rb') as f:
                models = pickle.load(f)
        biases = cls.get_biases(nn.neurons)
        weights = cls.get_weights(nn.connections)
        models[nn.model_name] = (biases, weights)
        with open('models.pkl', 'wb') as f:
            pickle.dump(models, f)
        print(f'Saving model: \"{nn.model_name}\"')

    @classmethod
    def load_model(cls, model_name):
        with open('models.pkl', 'rb') as f:
            models = pickle.load(f)
        shape = [len(layer) for layer in models[model_name][0]]
        neurons, connections = Architect.build(shape)
        cls.set_biases(neurons, models[model_name][0])
        cls.set_weights(connections, models[model_name][1])
        print(f'Loading model: \"{model_name}\"')
        return neurons, connections


class Neuron:

    def __init__(self):
        self.inc_conns = None
        self.out_conns = None
        self.activ_func = None
        self.deriv_func = None
        self.activation = 0
        self.bias = 0
        self.gradient = 0
        self.gradient_batch_sum = 0

    def compute_and_set_activation(self):
        inc_weights = [ic.weight for ic in self.inc_conns]
        inc_activs = [ic.source_neuron.activation for ic in self.inc_conns]
        weighted_sum = sum([w * a for w, a in zip(inc_weights, inc_activs)])
        self.activation = self.activ_func(weighted_sum + self.bias)

    def compute_and_update_gradient_values(self):
        out_weights = [oc.weight for oc in self.out_conns]
        out_cost_grd = [oc.target_neuron.gradient for oc in self.out_conns]
        weighted_sum = sum([w * e for w, e in zip(out_weights, out_cost_grd)])
        self.gradient = self.deriv_func(self.activation) * weighted_sum
        self.gradient_batch_sum += self.gradient

    def descend_bias_gradient(self, learning_rate):
        self.bias -= self.gradient_batch_sum * learning_rate


class Connection:

    def __init__(self):
        self.weight = 0
        self.source_neuron = None
        self.target_neuron = None

    def descend_weight_gradient(self, learning_rate):
        self.weight -= \
            self.source_neuron.activation * \
            self.target_neuron.gradient_batch_sum * \
            learning_rate


class Architect:

    @classmethod
    def build(cls, shape):
        neurons = cls.create_neurons(shape)
        connections = cls.create_connections(shape)
        cls.set_conn_references_in_neurons(neurons, connections)
        cls.set_neuron_references_in_conns(neurons, connections)
        cls.set_activation_and_derivative_functions_in_neurons(neurons)
        cls.initialize_weight_values_in_connections(connections)
        return neurons, connections

    @staticmethod
    def create_neurons(shape):
        neuron_layers = []
        for size in shape:
            neuron_layers.append([Neuron() for _ in range(size)])
        return neuron_layers

    @staticmethod
    def create_connections(shape):
        connection_layers = []
        for back, fore in zip(shape[:-1], shape[1:]):
            neuron_group = []
            for _ in range(back):
                neuron_group.append([Connection() for _ in range(fore)])
            connection_layers.append(neuron_group)
        return connection_layers

    @staticmethod
    def set_conn_references_in_neurons(neurons, connections):
        for neuron_layer, conn_layer in zip(neurons[1:-1], connections[1:]):
            for neuron, connection_group in zip(neuron_layer, conn_layer):
                neuron.out_conns = connection_group
        for neuron_layer, conn_layer in zip(neurons[1:], connections):
            for i, neuron in enumerate(neuron_layer):
                neuron.inc_conns = [c[i] for c in conn_layer]

    @staticmethod
    def set_neuron_references_in_conns(neurons, connections):
        zipped_layers = zip(neurons[:-1], connections, neurons[1:])
        for src_neuron_layer, conn_layer, tgt_neuron_layer in zipped_layers:
            for src_neuron, conn_group in zip(src_neuron_layer, conn_layer):
                for connection, tgt_neuron in zip(conn_group, tgt_neuron_layer):
                    connection.source_neuron = src_neuron
                    connection.target_neuron = tgt_neuron

    @staticmethod
    def set_activation_and_derivative_functions_in_neurons(neuron_layers):
        def relu_activation(x):
            return max(x, x * 0.01)
        def relu_derivative(x):
            return 1 if x > 0 else 0.01
        def identity(x):
            return x
        for neuron_layer in neuron_layers[1:-1]:
            for neuron in neuron_layer:
                neuron.activ_func = relu_activation
                neuron.deriv_func = relu_derivative
        for neuron in neuron_layers[-1]:
            neuron.activ_func = identity

    @staticmethod
    def initialize_weight_values_in_connections(conn_layers):
        def he_weight_init(n):
            return random.gauss(0, math.sqrt(2 / n))
        for conn_layer in conn_layers:
            neurons_in_prev_layer = len(conn_layer)
            for conn_group in conn_layer:
                for connection in conn_group:
                    connection.weight = he_weight_init(neurons_in_prev_layer)


class Evaluator:

    def __init__(self, neural_network, samples, num_of_samples):
        self.nn = neural_network
        self.samples = samples
        self.num_of_samples = num_of_samples
        self.accuracy = 0
        self.cost = 0

    def classified_correctly(self, sample):
        return sample['one_hot'] == self.nn.one_hot(sample['pixels'])

    def cost_function(self, sample):
        outputs = self.nn.output(sample['pixels'])
        targets = sample['one_hot']
        return sum([(o - t) ** 2 for o, t in zip(outputs, targets)])

    def print_report(self, start_time):
        elapsed = 'N/A'
        if start_time:
            elapsed = timedelta(seconds=round(time() - start_time))
        print(f'Acc: {round(self.accuracy, 3)}, '
              f'Cost: {round(self.cost, 3)}, '
              f'Elapsed Time: {elapsed}')

    def evaluate_network(self, start_time=None, print_report=False):
        accuracy_sum, cost_sum = 0, 0
        for _ in range(self.num_of_samples):
            sample = random.choice(self.samples)
            accuracy_sum += self.classified_correctly(sample)
            cost_sum += self.cost_function(sample)
        self.accuracy = accuracy_sum / self.num_of_samples
        self.cost = cost_sum / self.num_of_samples
        if print_report:
            self.print_report(start_time)
        return self.cost


class Trainer:

    def __init__(self, neural_network, evaluator, samples):
        self.nn = neural_network
        self.samples = samples
        self.ev = evaluator
        self.training_mins = None

    def zero_out_batch_gradient_sums(self):
        for layer in self.nn.neurons[1:-1]:
            for neuron in layer:
                neuron.gradient_batch_sum = 0

    def set_gradients_in_output_layer(self, sample):
        self.nn.output(sample['pixels'])
        for neuron, target in zip(self.nn.neurons[-1], sample['one_hot']):
            neuron.gradient = neuron.activation - target

    def backpropagate(self, sample):
        self.set_gradients_in_output_layer(sample)
        for layer in reversed(self.nn.neurons[1:-1]):
            for neuron in layer:
                neuron.compute_and_update_gradient_values()

    def gradient_descent(self, learning_rate):
        for neuron_layer in self.nn.neurons[1:]:
            for neuron in neuron_layer:
                neuron.descend_bias_gradient(learning_rate)
        for conn_layer in self.nn.connections:
            for conn_group in conn_layer:
                for connection in conn_group:
                    connection.descend_weight_gradient(learning_rate)

    def training_session(self, learning_rate, batch_size, session_iters):
        for _ in range(session_iters):
            self.zero_out_batch_gradient_sums()
            mini_batch = random.sample(self.samples, batch_size)
            for sample in mini_batch:
                self.backpropagate(sample)
            self.gradient_descent(learning_rate)

    def train_network(self, learning_rate, batch_size, session_iters):
        start_time = time()
        learning_rate /= batch_size
        session_iters = int(session_iters / batch_size)
        best_biases = Model.get_biases(self.nn.neurons)
        best_weights = Model.get_weights(self.nn.connections)
        lowest_cost = self.ev.evaluate_network(start_time, print_report=True)
        try:
            while True:
                self.training_session(learning_rate, batch_size, session_iters)
                cost = self.ev.evaluate_network(start_time, print_report=True)
                if cost < lowest_cost:
                    print('New Best!')
                    best_biases = Model.get_biases(self.nn.neurons)
                    best_weights = Model.get_weights(self.nn.connections)
                    lowest_cost = cost
                else:
                    Model.set_biases(self.nn.neurons, best_biases)
                    Model.set_weights(self.nn.connections, best_weights)
                if self.training_mins:
                    if (time() - start_time) / 60 < self.training_mins:
                        break
        except KeyboardInterrupt:
            print('Training stopped early by user')
        Model.set_biases(self.nn.neurons, best_biases)
        Model.set_weights(self.nn.connections, best_weights)
        Model.save_model(self.nn)


class Network:

    def __init__(self, model_name, shape=None):
        model = self.get_model(model_name, shape)
        self.neurons = model[0]
        self.connections = model[1]
        self.model_name = model_name
        self.features = None

    @staticmethod
    def get_model(model_name, shape):
        if shape:
            print('Creating new model')
            new_model = Architect.build(shape)
            return new_model
        else:
            loaded_model = Model.load_model(model_name)
            return loaded_model

    def set_features_in_input_layer(self):
        for neuron, feature in zip(self.neurons[0], self.features):
            neuron.activation = feature

    def feed_forward(self):
        for layer in self.neurons[1:]:
            for neuron in layer:
                neuron.compute_and_set_activation()

    def output(self, features):
        if self.features != features:
            self.features = features
            self.set_features_in_input_layer()
            self.feed_forward()
        return [n.activation for n in self.neurons[-1]]

    def classify(self, features):
        output = self.output(features)
        return output.index(max(output))

    def one_hot(self, features):
        classification = self.classify(features)
        return [float(i == classification) for i in range(10)]


def get_samples():
    with gzip.open('mnist_data.pkl.gz', 'rb') as f:
        data = pickle.load(f)
    return data['training_samples'], data['testing_samples']


if __name__ == '__main__':

    training_samples, testing_samples = get_samples()
    network = Network('model01', [28*28, 16, 16, 10])
    evaluator = Evaluator(network, testing_samples, 3000)
    trainer = Trainer(network, evaluator, training_samples)
    trainer.train_network(0.5, 16, 3000)
