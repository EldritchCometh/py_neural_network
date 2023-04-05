

import csv
import pickle


raw_training_data = []
raw_testing_data = []


with open('mnist_train.csv', newline='') as f:
    reader = csv.reader(f)
    next(reader)
    raw_training_data.extend(list(reader))


with open('mnist_test.csv', newline='') as f:
    reader = csv.reader(f)
    next(reader)
    raw_testing_data.extend(list(reader))


training_data = []
for data in raw_training_data:
    label = int(data[0])
    one_hot = [float(i == data[0]) for i in range(10)]
    pixels = [float(p)/255 for p in data[1:]]
    training_data.append(
        {'label': label,
         'one_hot': one_hot,
         'pixels': pixels})


testing_data = []
for data in raw_testing_data:
    label = int(data[0])
    one_hot = [float(i == data[0]) for i in range(10)]
    pixels = [float(p)/255 for p in data[1:]]
    testing_data.append(
        {'label': label,
         'one_hot': one_hot,
         'pixels': pixels})


mnist_data = {
    'training_samples': training_data,
    'testing_samples': testing_data
}


with open('mnist_data.pkl', 'wb') as f:
    pickle.dump(mnist_data, f)

