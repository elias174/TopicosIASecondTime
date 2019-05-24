import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def read_and_create_data(filename):
    df = pd.read_csv(filename)
    df2 = df.join(pd.get_dummies(df.pop('species')))

    _Y = df2[df2.columns[-3:]]
    _X = df2[df2.columns[:4]]

    return _X, _Y


def plot_elements(_X, _y):
    one_type = _X[_y['Iris-setosa'] == 1]
    second_type = _X[_y['Iris-versicolor'] == 1]
    third_type = _X[_y['Iris-virginica'] == 1]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(one_type[one_type.columns[0]],
                one_type[one_type.columns[1]],
                one_type[one_type.columns[2]])
    ax.scatter(second_type[second_type.columns[0]],
                second_type[second_type.columns[1]],
                second_type[second_type.columns[2]])
    ax.scatter(third_type[third_type.columns[0]],
                third_type[third_type.columns[1]],
                third_type[third_type.columns[2]])

    # plt.scatter(second_type[:, 1], second_type[:, 2], s=10)
    plt.show()


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


class HiddenLayer(object):
    def __init__(self, n_input, n_activations, function_activation,
                 last_layer=False, W=None, b=None):
        self.n_input = n_input
        self.n_activations = n_activations
        self.function_activation = function_activation
        self.W = W
        self.b = b
        self.last_layer = last_layer
        if not self.W:
            # Random Weights
            self.W = np.random.rand(n_activations,n_input)
        if not self.b:
            self.b = np.array([1.0])

    def get_activations(self, input):
        if not self.last_layer:
            self.activations = np.append(
                self.b, self.function_activation(np.matmul(input, self.W.T)))
        else:
            self.activations = self.function_activation(
                np.matmul(input, self.W.T))

        return self.activations


class MLP(object):
    def __init__(self, n_hidden_layers, n_activations, n_input):
        self.n_hidden_layers = n_hidden_layers
        self.n_activations = n_activations
        self.n_input = n_input + 1
        self.function_activation = sigmoid
        # Add first layer
        self.hidden_layers = [
            HiddenLayer(
                self.n_input,
                self.n_activations,
                self.function_activation
            )
        ]
        for i in range(1, self.n_hidden_layers):
            layer_to_append = HiddenLayer(
                # how n_input use the size of columns of
                # the weights in the last layer
                self.hidden_layers[i-1].W.shape[1],
                self.n_activations,
                self.function_activation
            )
            self.hidden_layers.append(layer_to_append)

    def forward(self, input):
        input = np.append(np.array([1.0]), input)
        last_activation = input
        # Evaluate in all hidden layers
        for hidden_layer in self.hidden_layers:
            last_activation = hidden_layer.get_activations(last_activation)

        # Process last layer
        last_layer = HiddenLayer(
            last_activation.shape[0],
            3,
            self.function_activation,
            last_layer=True
        )

        last_activation = last_layer.get_activations(last_activation)
        return last_activation


_X, _y = read_and_create_data('data/iris.data')
# _X = np.append(np.array([1.0]), _X.iloc[0])
_X = _X.iloc[0]

# layer = HiddenLayer(_X, _X.shape[0], 3, sigmoid)
# print(layer.get_activations())

mlp = MLP(3,4, _X.shape[0])
print(mlp.forward(_X))