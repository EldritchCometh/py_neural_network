import gzip
import math
import os
import pickle
import random
from time import time
from datetime import datetime, timedelta


class IO:

    @staticmethod
    def get_samples():
        with gzip.open('mnist_data.pkl.gz', 'rb') as f:
            data = pickle.load(f)
        return data['training_samples'], data['testing_samples']

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
    def save_model(cls, nn, model_name):
        models = {}
        if os.path.isfile('models.pkl'):
            with open('models.pkl', 'rb') as f:
                models = pickle.load(f)
        biases = cls.get_biases(nn.neurons)
        weights = cls.get_weights(nn.connections)
        models[model_name] = (biases, weights)
        with open('models.pkl', 'wb') as f:
            pickle.dump(models, f)

    @classmethod
    def load_model(cls, model_name):
        with open('models.pkl', 'rb') as f:
            models = pickle.load(f)
        shape = [len(layer) for layer in models[model_name][0]]
        neurons, connections = Architect.build(shape)
        cls.set_biases(neurons, models[model_name][0])
        cls.set_weights(connections, models[model_name][1])
        return neurons, connections


class Neuron:

    def __init__(self):
        self.inc_conns = None
        self.out_conns = None
        self.activ_func = None
        self.deriv_func = None
        self.activation = 0
        self.bias = 0
        self.error_gradient = 0
        self.batch_error_gradients_sum = 0

    def compute_and_set_activation(self):
        inc_weights = [ic.weight for ic in self.inc_conns]
        inc_activs = [ic.source_neuron.activation for ic in self.inc_conns]
        weighted_sum = sum([w * a for w, a in zip(inc_weights, inc_activs)])
        self.activation = self.activ_func(weighted_sum + self.bias)

    def compute_and_update_error_gradient(self):
        out_weights = [oc.weight for oc in self.out_conns]
        out_err_grd = [oc.target_neuron.error_gradient for oc in self.out_conns]
        weighted_sum = sum([w * e for w, e in zip(out_weights, out_err_grd)])
        self.error_gradient = self.deriv_func(self.activation) * weighted_sum
        self.batch_error_gradients_sum += self.error_gradient

    def descend_bias_gradient(self, learning_rate):
        self.bias -= self.batch_error_gradients_sum * learning_rate


class Connection:

    def __init__(self):
        self.weight = 0
        self.source_neuron = None
        self.target_neuron = None

    def descend_weight_gradient(self, learning_rate):
        self.weight -= \
            self.source_neuron.activation * \
            self.target_neuron.batch_error_gradients_sum * \
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
            return max(0, x)
        def relu_derivative(x):
            return 1 if x > 0 else 0
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


class Network:

    def __init__(self, nn):
        self.neurons, self.connections = nn
        self.features = None

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


class Evaluator:

    def __init__(self, neural_network, samples):
        self.nn = neural_network
        self.samples = samples
        self.init_report = self.evaluate_network(1000)
        self.last_report = 0
        self.report_freq = 10
        self.basic_report_samples = 100
        self.final_report_samples = 1000

    def classified_accurately(self, samples):
        return samples['one_hot'] == self.nn.one_hot(samples['pixels'])

    def cost_function(self, samples):
        outputs = self.nn.output(samples['pixels'])
        targets = samples['one_hot']
        return sum([(o - t) ** 2 for o, t in zip(outputs, targets)])

    def evaluate_network(self, num_of_samples):
        test = time()
        accuracy_sum = 0
        cost_sum = 0
        for _ in range(num_of_samples):
            sample = random.choice(self.samples)
            accuracy_sum += self.classified_accurately(sample)
            cost_sum += self.cost_function(sample)
        accuracy_average = accuracy_sum / num_of_samples
        cost_average = cost_sum / num_of_samples
        return accuracy_average, cost_average

    def basic_report(self, start_time):
        mse = None
        if self.report_freq and time() - self.last_report > self.report_freq:
            self.last_report = time()
            acc, mse = self.evaluate_network(self.basic_report_samples)
            elapsed = timedelta(seconds=round(time() - start_time))
            print(f'Acc: {round(acc, 3)}, '
                  f'MSE: {round(mse, 3)}, '
                  f'Elapsed Time: {elapsed}')
        return mse

    def final_report(self, learning_rate, batch_size, start_time):
        init_acc, init_mse = self.init_report
        final_acc, final_mse = self.evaluate_network(self.final_report_samples)
        start = datetime.fromtimestamp(start_time).strftime("%H:%M:%S")
        end = datetime.fromtimestamp(time()).strftime("%H:%M:%S")
        elapsed = timedelta(seconds=round(time() - start_time))
        print('#####################  -  Final Report -  #####################')
        print(f'Start Time: {start},',
              f'End Time: {end},',
              f'Elapsed Time: {elapsed}\n'
              f'Learning Rate: {round(learning_rate, 3)},',
              f'Batch Size: {batch_size},',
              f'Adj Learning Rate: {round((learning_rate / batch_size), 5)}\n'
              f'Pre-training Accuracy:  {round(init_acc, 3)},',
              f'Pre-training MSE:  {round(init_mse, 3)}\n'
              f'Post-training Accuracy: {round(final_acc, 3)},',
              f'Post-training MSE: {round(final_mse, 3)}')
        print('###############################################################')
        return final_mse


class Trainer:

    def __init__(self, nn, samples, ev=None, name=None):
        self.nn = nn
        self.samples = samples
        self.ev = ev
        self.name = name

    def zero_out_batch_error_gradient_sums(self):
        for layer in self.nn.neurons[1:-1]:
            for neuron in layer:
                neuron.batch_error_gradients_sum = 0

    def set_error_gradients_in_output_layer(self, sample):
        self.nn.output(sample['pixels'])
        for neuron, target in zip(self.nn.neurons[-1], sample['one_hot']):
            neuron.error_gradient = neuron.activation - target

    def backpropagate(self):
        for layer in reversed(self.nn.neurons[1:-1]):
            for neuron in layer:
                neuron.compute_and_update_error_gradient()

    def backpropagation(self, mini_batch):
        self.zero_out_batch_error_gradient_sums()
        for sample in mini_batch:
            self.set_error_gradients_in_output_layer(sample)
            self.backpropagate()

    def gradient_descent(self, learning_rate):
        # Adjust biases by gradient descent
        for neuron_layer in self.nn.neurons[1:]:
            for neuron in neuron_layer:
                neuron.descend_bias_gradient(learning_rate)
        # Adjust weights by gradient descent
        for conn_layer in self.nn.connections:
            for conn_group in conn_layer:
                for connection in conn_group:
                    connection.descend_weight_gradient(learning_rate)

    def train_network(self, learning_rate=0.01, batch_size=1, mins=1.0):
        start_time = time()
        max_mse = 99999
        try:
            while (time() - start_time) / 60 < mins:
                mini_batch = random.sample(self.samples, batch_size)
                self.backpropagation(mini_batch)
                self.gradient_descent(learning_rate / batch_size)
                if self.ev:
                    mse = self.ev.basic_report(start_time)
                    if mse and mse < max_mse:
                        print('saving')
                        IO.save_model(self.nn, self.name)
                        max_mse = mse
        except KeyboardInterrupt:
            print('Training stopped early by user.')


if __name__ == '__main__':

    training_samples, testing_samples = IO.get_samples()
    # structure = Architect.build([784, 16, 16, 10])
    structure = IO.load_model('model01')
    network = Network(structure)
    evaluator = Evaluator(network, testing_samples)
    evaluator.report_freq = 20
    evaluator.basic_report_samples = 1000
    trainer = Trainer(network, training_samples, evaluator, 'model01')
    trainer.train_network(learning_rate=0.005, batch_size=1, mins=5)
