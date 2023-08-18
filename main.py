import numpy as np
from models.model import Model
from layers.linear import Linear


def create_model():
    inputs = Linear(input_size=2, output_size=6, activation='relu')
    # x = Linear(output_size=4, activation='relu')(inputs)
    # outputs = Linear(output_size=1, activation='relu')(x)
    outputs = Linear(output_size=2, activation='softmax')(inputs)

    model = Model(inputs, [outputs])
    return model


# training data
# x_train = np.array([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]])
# y_train = np.array([[[0]], [[1]], [[1]], [[0]]])
x_train = np.array([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]])
y_train = np.array([[[1, 0]], [[0, 1]], [[0, 1]], [[1, 0]]])


model = create_model()
# model.compile(loss_fn='mse')
model.compile(loss_fn='mse')
loss = model.fit(x_train,
                 y_train,
                 epochs=500,
                 learning_rate=0.03,
                 print_loss=False)

print(f"Loss: {round(loss, 4)}")

# test
out = np.array(model.predict(x_train))
print(out)
