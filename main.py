"""Main - build a model, train it, test it."""


import numpy as np
import matplotlib.pyplot as plt
import math
from models.model import Model
from layers.dense import Dense
from layers.input import Input
from optimizers.momentum import Momentum
from optimizers.sgd import SGD


def create_model():
    """Define the layers of the model."""
    inputs = Input(shape=(3, ))
    x = Dense(output_size=16,
              activation='tanh',
              weights_initializer='xavier_uniform')(inputs)
    x = Dense(output_size=8,
              activation='relu',
              weights_initializer='xavier_normal')(x)
    x = Dense(output_size=4,
              activation='relu',
              weights_initializer='xavier_uniform')(x)
    outputs = Dense(output_size=1,
                    activation='relu',
                    weights_initializer='xavier_normal')(x)

    return Model(inputs, outputs)


def create_dataset(size=100):
    """Create a dataset using for a function that the model should learn."""
    def function(x1, x2, x3):
        return abs(math.sin(0.8 * x1 ** 2 + 0.2 * x2 ** 3 - 0.5 * math.cos(x3)))

    features = np.random.randn(size, 3)
    labels = np.array([[function(*x)] for x in features])

    return features, labels


def train_test_split(X, y, split=0.7):
    """Split the dataset into training and testing sets."""
    n = int(len(X) * split)
    features_train = X[:n]
    labels_train = y[:n]
    features_test = X[n:]
    labels_test = y[n:]
    return features_train, labels_train, features_test, labels_test


np.random.seed(1)

X, y = create_dataset(size=200)
X_train, y_train, X_test, y_test = train_test_split(X, y)

model = create_model()
model.compile(loss_fn='mse', optimizer=SGD())
loss = model.fit(X_train,
                 y_train,
                 epochs=10,
                 batch_size=4,
                 print_loss=True)

print(f"Loss: {round(loss, 8)}")

# test
out = np.array(model.predict(X_test))

for i in range(len(out)):
    plt.plot([i, i], [out[i], y_test[i][0]], 'ro-')
plt.show()