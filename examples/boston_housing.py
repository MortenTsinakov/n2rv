"""
Example of using the library on Boston housing dataset.

The Boston housing dataset can be found here:
https://www.kaggle.com/datasets/altavish/boston-housing-dataset?resource=download
"""

import sys
import os
import pandas as pd

# Add the project directory to the Python path
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_dir)

from models.model import Model
from layers.input import Input
from layers.dense import Dense
from optimizers.adam import Adam
from metrics.mae import MAE


def import_data(filename: str) -> pd.DataFrame:
    """Import the dataset."""
    return pd.read_csv(filename)


def fill_nan_values_with_mean(data: pd.DataFrame) -> pd.DataFrame:
    """Replace NaN values with the mean of the column."""
    means = data.mean()
    return data.fillna(means)


def normalize_values(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize the values in each column using min-max scaling."""
    return (data - data.min()) / (data.max() - data.min())


def train_test_split(dataset: pd.DataFrame, split: float = 0.8) -> tuple:
    """Split dataset into training and testing set."""
    n_train = int(len(dataset) * split)
    y_tr = dataset["MEDV"].iloc[:n_train].to_numpy().reshape(-1, 1)
    y_te = dataset["MEDV"].iloc[n_train:].to_numpy().reshape(-1, 1)
    x_tr = dataset.drop(columns=["MEDV"]).iloc[:n_train].to_numpy()
    x_te = dataset.drop(columns=["MEDV"]).iloc[n_train:].to_numpy()
    return x_tr, y_tr, x_te, y_te


def get_data(filename: str, random_state: int = None) -> tuple:
    """Import and prepare data for training."""
    data = import_data(filename=filename)
    data = fill_nan_values_with_mean(data)
    label_min, label_max = data["MEDV"].min(), data["MEDV"].max()
    data = normalize_values(data)
    data = data.sample(frac=1.0, random_state=random_state)
    return data, label_min, label_max


def get_model() -> Model:
    """Build the model."""
    inputs = Input(shape=(13, ))
    x = Dense(output_size=64,
              activation='relu',
              weights_initializer='he_normal')(inputs)
    x = Dense(output_size=128,
              activation='relu',
              weights_initializer='he_normal')(x)
    x = Dense(output_size=64,
              activation='relu',
              weights_initializer='he_normal')(x)
    outputs = Dense(1,
                    activation='tanh',
                    weights_initializer='xavier_normal')(x)
    return Model(inputs=inputs, outputs=outputs)


if __name__ == "__main__":
    filename = "../datasets/HousingData.csv"
    data, label_min, label_max = get_data(filename=filename)
    x_train, y_train, x_test, y_test = train_test_split(data, split=0.75)

    model = get_model()
    model.compile(loss_fn="mse",
                  optimizer=Adam(),
                  metrics=[MAE()])
    loss = model.fit(x_train=x_train,
                     y_train=y_train,
                     epochs=500,
                     print_metrics=False,
                     batch_size=32)

    print()
    print("Evaluation on test data")
    print("-----------------------")
    evaluation = model.evaluate(x_test=x_test, y_test=y_test)
    preds = model.predict(x_test)
    for k, v in evaluation.items():
        print(k, v)

    print()
    mae = evaluation["Mean Absolute Error"]
    denormalized_error = mae * (label_max - label_min) * 1000
    print(f"Denormalized average prediciton error: ${round(denormalized_error, 2)}")
