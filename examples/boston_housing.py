"""
Example of using the library on Boston housing dataset.

To run this file, create a dataset directory (called 'datasets') one directory up from this file and
add Boston housing dataset with the name 'HousingData.csv' in that directory.

The Boston housing dataset can be found here:
https://www.kaggle.com/datasets/altavish/boston-housing-dataset?resource=download
"""

from random import randint
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Add the project directory to the Python path
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_dir)

from models.model import Model
from layers.input import Input
from layers.dense import Dense
from optimizers.adam import Adam


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


def get_data(filename: str, random_state: int = 42) -> tuple:
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
              weights_initializer='he_uniform')(inputs)
    x = Dense(output_size=32,
              activation='relu',
              weights_initializer='he_uniform')(x)
    outputs = Dense(1,
                    activation='sigmoid',
                    weights_initializer='xavier_normal')(x)
    return Model(inputs=inputs, outputs=outputs)


def plot_the_comparison(pred, true):
    """Plot the predictions and the true values to compare them."""
    x = np.arange(len(pred))

    width = 0.5

    plt.figure(figsize=(10, 6))
    plt.bar(x - width / 2, pred, width, color='red', alpha=0.5, label='Predicted')
    plt.bar(x - width / 2, true, width, color='green', alpha=0.5, label='Actual')

    plt.xticks([])
    plt.xlabel('Data points')
    plt.ylabel('Median value of owner-occupied homes in dollars.')
    plt.title('Comparison of Predicted vs Actual Values')
    plt.legend()

    plt.grid(False)
    plt.show()


def count_errors(pred, true):
    """Count the prediction errors by range."""
    error_counts = {
        "<1000": 0,
        "1000-2000": 0,
        "2000-3000": 0,
        "3000-4000": 0,
        "4000-5000": 0,
        ">5000": 0
    }
    max_error = float('-inf')
    min_error = float('inf')

    for p, t in zip(pred, true):
        diff = abs(p[0] - t[0])
        if diff > max_error:
            max_error = diff
        if diff < min_error:
            min_error = diff
        if diff < 1000:
            error_counts['<1000'] += 1
        elif 1000 <= diff < 2000:
            error_counts['1000-2000'] += 1
        elif 2000 <= diff < 3000:
            error_counts['2000-3000'] += 1
        elif 3000 <= diff < 4000:
            error_counts['3000-4000'] += 1
        elif 4000 <= diff < 5000:
            error_counts['4000-5000'] += 1
        else:
            error_counts['>5000'] += 1

    for k, v in error_counts.items():
        print(f"Predictions with error {k}: {v}")
    print()
    print(f"Min error: {round(min_error, 2)}")
    print(f"Max error: {round(max_error, 2)}")


if __name__ == "__main__":
    filename = "../datasets/HousingData.csv"
    data, label_min, label_max = get_data(filename=filename, random_state=randint(0, 100))
    x_train, y_train, x_test, y_test = train_test_split(data)

    model = get_model()
    model.compile(loss_fn="mse",
                  optimizer=Adam())
    loss = model.fit(x_train=x_train,
                     y_train=y_train,
                     epochs=200,
                     print_metrics=False,
                     batch_size=32)
    print(f"Final loss: {loss}")

    # Scale predictions and actual results back and because they are
    # thouseands in original dataset multiply with 1000
    preds = model.predict(x_test)
    preds = (preds * (label_max - label_min ) + label_min) * 1000
    y_test = (y_test * (label_max - label_min) + label_min) * 1000

    count_errors(preds, y_test)
    plot_the_comparison(preds.reshape(-1), y_test.reshape(-1))
