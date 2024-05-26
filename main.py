"""Testing the library."""

import numpy as np

from n2rv.layers.dense import Dense
from n2rv.layers.input import Input
from n2rv.models.model import Model
from n2rv.optimizers.adam import Adam
from n2rv.metrics.binary_accuracy import BinaryAccuracy
from n2rv.metrics.precision import Precision


def get_label(data: list) -> int:
    """Produce label for a sample."""
    x1, x2, x3 = data
    answer = x1**2 + x2 - x3
    if answer < 0:
        return 0
    return 1


def train_test_split(data: np.ndarray,
                     labels: np.ndarray,
                     split: float = 0.7) -> tuple:
    """Split data into training and testing set."""
    n = int(len(data) * split)
    x_tr = data[:n]
    y_tr = labels[:n]
    x_te = data[n:]
    y_te = labels[n:]
    return x_tr, y_tr, x_te, y_te


def get_model() -> Model:
    """Create a model."""
    inputs = Input((3,))
    x = Dense(output_size=8, activation="tanh",
              weights_initializer="xavier_uniform")(
        inputs
    )
    x = Dense(output_size=4, activation="tanh",
              weights_initializer="xavier_uniform")(x)
    outputs = Dense(
        output_size=1, activation="sigmoid",
        weights_initializer="xavier_uniform"
    )(x)

    return Model(inputs=inputs, outputs=outputs)


features = np.random.randn(500, 3)
labels = np.array([get_label(sample) for sample in features]).reshape(-1, 1)

x_train, y_train, x_test, y_test = train_test_split(features, labels)

model = get_model()
model.compile(loss_fn="binary_cross_entropy",
              optimizer=Adam(),
              metrics=[BinaryAccuracy(), Precision()])
loss = model.fit(
    x_train=x_train,
    y_train=y_train,
    epochs=200,
    print_metrics=True,
    batch_size=32
)
