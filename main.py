import numpy as np
from models.model import Model
from layers.linear import Linear


def create_model():
    inputs = Linear(input_size=2, output_size=8, activation='tanh')
    x = Linear(output_size=4, activation='sigmoid')(inputs)
    outputs = Linear(output_size=1, activation='relu')(x)

    model = Model(inputs, [outputs])
    return model


# training data
x_train = np.array([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

model = create_model()
model.compile(loss_fn='mse')
loss = model.fit(x_train,
                 y_train,
                 epochs=500,
                 learning_rate=0.1,
                 print_loss=True)

print(f"Loss: {round(loss, 4)}")

# test
out = model.predict(x_train)
print(out)
