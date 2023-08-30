import numpy as np
from models.model import Model
from layers.dense import Dense
from layers.input import Input


def create_model():
    inputs = Input(shape=(1, 2))
    x = Dense(output_size=8, activation='tanh')(inputs)
    x = Dense(output_size=4, activation='relu')(x)
    outputs = Dense(output_size=1, activation='sigmoid')(x)

    model = Model(inputs, outputs)
    return model


# training data
x_train = np.array([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])
# x_train = np.array([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]])
# y_train = np.array([[[1, 0]], [[0, 1]], [[0, 1]], [[1, 0]]])


model = create_model()
model.compile(loss_fn='mse')
loss = model.fit(x_train,
                 y_train,
                 epochs=500,
                 learning_rate=0.1,
                 print_loss=False)

print(f"Loss: {round(loss, 4)}")

# test
out = np.array(model.predict(x_train))
print(np.round(out, 2))
