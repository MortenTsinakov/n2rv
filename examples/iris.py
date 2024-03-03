"""
Example of using the library on iris dataset.

The iris dataset can be downloaded from:
https://archive.ics.uci.edu/dataset/53/iris
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
from metrics.accuracy import Accuracy


def import_data(filename: str) -> pd.DataFrame:
    """Import the dataset."""
    names = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
        "species"
    ]
    return pd.read_csv(filename, names=names)


def species_to_categorical(dataset: pd.DataFrame) -> None:
    """Turn the species column from string to integer."""
    categories = dataset["species"].unique()
    mapping = {}
    for idx, cat in enumerate(categories):
        mapping[cat] = idx
    dataset["species"] = dataset["species"].map(mapping).astype(int)


def normalize_column(column):
    """Normalize column using min max normalization."""
    min_val = column.min()
    max_val = column.max()
    return (column - min_val) / (max_val - min_val)


def get_data(random_state: int = 42):
    """
    Import iris data, change 'species' values to numerical categories,
    normalize and shuffle the data.
    """
    dataset = import_data('../datasets/iris.data')
    species_to_categorical(dataset)
    features = dataset.select_dtypes(include=['float64']).columns
    dataset[features] = dataset[features].apply(normalize_column)
    dataset = dataset.sample(frac=1.0, random_state=random_state)
    return dataset


def train_test_split(dataset: pd.DataFrame, split: float = 0.7) -> tuple:
    """Split the data into training and testing set."""
    n_train = int(len(dataset) * split)
    # Get all rows until n_train of column 'species' and turn them into one-hot encoded vectors
    y_tr = dataset['species'].iloc[:n_train]
    y_tr = pd.get_dummies(y_tr, ['species']).to_numpy()
    # Get all rows from n_train of column 'species' and turn them into one-hot encoded vectors
    y_te = dataset['species'].iloc[n_train:]
    y_te = pd.get_dummies(y_te, ['species']).to_numpy()
    x_tr = dataset.drop(columns=['species']).iloc[:n_train].to_numpy()
    x_te = dataset.drop(columns=['species']).iloc[n_train:].to_numpy()
    return x_tr, y_tr, x_te, y_te


def get_model() -> Model:
    """Build the model."""
    inputs = Input(shape=(4, ))
    x = Dense(output_size=32,
              activation='relu',
              weights_initializer='he_normal',)(inputs)
    x = Dense(output_size=16,
              activation='relu',
              weights_initializer='he_normal')(x)
    outputs = Dense(output_size=3,
                    activation='softmax',
                    weights_initializer='xavier_normal')(x)

    return Model(inputs=inputs, outputs=outputs)


if __name__ == "__main__":
    data = get_data(random_state=80)
    x_train, y_train, x_test, y_test = train_test_split(data)

    model = get_model()
    model.compile(loss_fn='categorical_cross_entropy',
                optimizer=Adam(),
                metrics=[Accuracy(decimal_places=4)])
    model.fit(x_train=x_train,
              y_train=y_train,
              epochs=100,
              print_metrics=True,
              batch_size=32)

    print()
    print("Evaluation on test data")
    print("-----------------------")
    evaluation = model.evaluate(x_test=x_test, y_test=y_test)
    for k, v in evaluation.items():
        print(k, v)
