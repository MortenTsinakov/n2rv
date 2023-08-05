import numpy as np
from model import Model
from linear import Linear
from activation import Activation

# training data
x_train = np.array([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

# Build model
model = Model(
    [
        Linear(2, 3),
        Activation('relu'),
        Linear(3, 1),
        Activation('relu'),
    ]
)

# train
model.use_loss_function('mse')
model.fit(x_train, y_train, epochs=1000, learning_rate=0.1)

# test
out = model.predict(x_train)
print(out)
