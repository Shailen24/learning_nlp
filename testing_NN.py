import torch
import numpy as np
from torchvision import datasets, transforms
from neural_network import NeuralNetwork
import pickle


training_size = 500

train_set = datasets.FashionMNIST('./data', train=True, download=False)
test_set = datasets.FashionMNIST('./data', train=False, download=False)
train_set_array = train_set.data.numpy()[:training_size]
test_set_array = test_set.data.numpy()[:training_size]
train_labels = train_set.targets.numpy()[:training_size]

num_classes = 10
# Perform one-hot encoding on the labels
train_labels_onehot = torch.nn.functional.one_hot(torch.from_numpy(train_labels), num_classes=num_classes).numpy()

# Flatten training data
flattened_train = np.zeros((len(train_set_array), len(train_set_array[0, :]) * len(train_set_array[0, 0])))
for i in range(len(train_set_array)):
    flattened_train[i, :] = train_set_array[i, :, :].flatten()

X = flattened_train.T
Y = train_labels_onehot

with open('trained_nn_500n_50iter.pickle', 'rb') as handle:
    trained_NN = pickle.load(handle)

outputs = trained_NN.forward_propagation(X)[0]  # (250, 10)
max_outputs = np.argmax(outputs, axis=1)    # (250, 1)
print(np.sum(max_outputs == train_labels))


