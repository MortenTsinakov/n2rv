"""Main - build a model, train it, test it."""


import numpy as np
import matplotlib.pyplot as plt
from models.model import Model
from layers.dense import Dense
from layers.input import Input
from optimizers.momentum import Momentum
from optimizers.sgd import SGD


def create_model():
    """Define the layers of the model."""
    inputs = Input(shape=(2, ))
    x = Dense(output_size=4,
              activation='tanh',
              weights_initializer='xavier_uniform')(inputs)
    outputs = Dense(output_size=1,
                    activation='linear',
                    weights_initializer='he_uniform')(x)

    return Model(inputs, outputs)


def create_dataset(size=100):
    """Create a dataset using for a function that the model should learn."""
    def function(x1, x2):
        return 0.3 * (x1 ** 2) + 0.2 * x2 + 0.5

    features = np.random.randn(size, 2)
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


np.random.seed(42)

X, y = create_dataset(size=100)
X_train, y_train, X_test, y_test = train_test_split(X, y)

model = create_model()
model.compile(loss_fn='mse', optimizer=Momentum(learning_rate=0.1))
loss = model.fit(X_train,
                 y_train,
                 epochs=200,
                 batch_size=8,
                 print_loss=False)

print(f"Loss: {round(loss, 8)}")

# test
out = np.array(model.predict(X_test))

plt.plot(out)
plt.plot(y_test)
plt.show()
