import numpy as np
from models.model import Model
from layers.dense import Dense
from layers.input import Input
import matplotlib.pyplot as plt


def create_model():
    inputs = Input(shape=(2))
    x = Dense(output_size=4, activation='tanh')(inputs)
    outputs = Dense(output_size=1, activation='linear')(x)

    model = Model(inputs, outputs)
    return model


def create_dataset(size=100):
    def function(x1, x2):
        # return 0.3 * (x1 ** 2) + 0.5 * x2 + 0.1
        return 0.3 * (x1 ** 2) + 0.2 * x2 + 0.5

    X = np.random.randn(size, 2)
    y = np.array([[function(*x)] for x in X])

    return X, y


def train_test_split(X, y, split=0.7):
    n = int(len(X) * split)
    X_train = X[:n]
    y_train = y[:n]
    X_test = X[n:]
    y_test = y[n:]
    return X_train, y_train, X_test, y_test


np.random.seed(12)
X, y = create_dataset(size=100)
X_train, y_train, X_test, y_test = train_test_split(X, y)

model = create_model()
model.compile(loss_fn='mse')
loss = model.fit(X_train,
                 y_train,
                 epochs=100,
                 learning_rate=0.1,
                 batch_size=1,
                 print_loss=False)

print(f"Loss: {round(loss, 4)}")

# test
out = np.array(model.predict(X_test))

plt.plot(out)
plt.plot(y_test)
plt.show()
