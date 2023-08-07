import numpy as np
from model import Model
from linear import Linear


def create_model():
    inputs = Linear(input_size=2, output_size=3, activation='tanh')
    outputs = Linear(output_size=1, activation='tanh')(inputs)

    print(inputs.weights.shape)
    print(outputs.weights.shape)

    model = Model(inputs, [outputs])
    return model


# training data
x_train = np.array([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

model = create_model()
model.use_loss_function('mse')
model.compile()
model.fit(x_train, y_train, epochs=1000, learning_rate=0.1)

# test
out = model.predict(x_train)
print(out)
