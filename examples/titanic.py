"""
Example of using the library on Titanic dataset.

The Titanic dataset can be downloaded from:
https://www.kaggle.com/datasets/yasserh/titanic-dataset
"""

import os
import sys

import pandas as pd

# Add the project directory to the Python path
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_dir)

from n2rv.layers.dense import Dense
from n2rv.layers.input import Input
from n2rv.models.model import Model
from n2rv.optimizers.adam import Adam
from n2rv.metrics.binary_accuracy import BinaryAccuracy
from n2rv.metrics.precision import Precision


def import_data(filename: str) -> pd.DataFrame:
    """
    Import the dataset

    inputs:
        filename (str) - the name of the dataset file
    return:
        Pandas DataFrame object
    """
    return pd.read_csv(filename)


def drop_unnecessary_columns(data: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns that contain unnecessary information like id and person's name
    and columns that contain too little information to be useful like Cabin.

    inputs:
        data (pd.DataFrame) - imported data
    return:
        Pandas DataFrame object with unnecessary columns dropped
    """
    return data.drop(["PassengerId", "Name", "Cabin", "Ticket", "Embarked"],
                     axis=1)


def impute_missing_values_with_mean(data: pd.DataFrame) -> pd.DataFrame:
    """
    Replace NaN values with column's mean value.

    inputs:
        data (pd.DataFrame) - imported data
    return:
        Pandas DataFrame object with NaN values replaced by column's mean.
    """
    data["Age"] = data["Age"].fillna(data["Age"].mean())
    return data


def categorize_string_values(data: pd.DataFrame) -> pd.DataFrame:
    """
    Replace string values with numerical categories.

    inputs:
        data (pd.DataFrame) - imported data
    return:
        Pandas DataFrame object where string values have been replaced with
            numerical categories
    """
    categories = data["Sex"].unique()
    mapping = {}
    for idx, cat in enumerate(categories):
        mapping[cat] = idx
    data["Sex"] = data["Sex"].map(mapping).astype(int)
    return data


def normalize_values(data: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize values to range (0, 1)

    inputs:
        data (pd.DataFrame) - imported data
    return:
        Pandas DataFrame object with normalized column values
    """
    cols = ["Age", "SibSp", "Parch", "Fare"]
    data[cols] = (data[cols] - data[cols].min()) /\
        (data[cols].max() - data[cols].min())
    return data


def get_data(filename: str, random_state: float = 42) -> pd.DataFrame:
    """
    Import, modify and return Titanic dataset.

    inputs:
        filename (str) - the name of the dataset file
        random_state (int) - seed for randomizing the order of samples
    return:
        Pandas DataFrame object with modified data and shuffled order
    """
    data = import_data(filename)
    data = drop_unnecessary_columns(data)
    data = impute_missing_values_with_mean(data)
    data = categorize_string_values(data)
    data = normalize_values(data)
    data.sample(frac=1.0, random_state=random_state)
    return data


def train_test_split(data: pd.DataFrame, split: float = 0.7) -> tuple:
    """
    Split the data into training and testing sets.

    inputs:
        data (pd.DataFrame) - imported data
        split (float) - size of the training set
    return:
        tuple of training features, trainging labels, testing features and
            testing labels
    """
    n = int(len(data) * split)
    x_train = data.drop(["Survived"], axis=1)[:n].to_numpy()
    y_train = data["Survived"][:n].to_numpy().reshape(-1, 1)
    x_test = data.drop(["Survived"], axis=1)[n:].to_numpy()
    y_test = data["Survived"][n:].to_numpy().reshape(-1, 1)

    return x_train, y_train, x_test, y_test


def get_model() -> Model:
    """
    Define the model layers and build the model.

    returns:
        Model object
    """
    inputs = Input(shape=(6,))
    x = Dense(
        output_size=64,
        activation="relu",
        weights_initializer="he_normal")(inputs)
    outputs = Dense(
        output_size=1,
        activation="sigmoid",
        weights_initializer="xavier_normal"
    )(x)

    return Model(inputs=inputs, outputs=outputs)


if __name__ == "__main__":
    filename = "../datasets/Titanic-Dataset.csv"
    data = get_data(filename=filename, random_state=42)

    x_train, y_train, x_test, y_test = train_test_split(data, split=0.7)

    model = get_model()
    model.compile(loss_fn="binary_cross_entropy",
                  optimizer=Adam(),
                  metrics=[BinaryAccuracy(), Precision()])
    loss = model.fit(
        x_train=x_train,
        y_train=y_train,
        epochs=200,
        print_metrics=False,
        batch_size=32
    )
    print(f"Final training loss: {loss}")

    print()
    print("Evaluation on test data")
    print("-----------------------")
    evaluation = model.evaluate(x_test=x_test, y_test=y_test)

    for metric, result in evaluation.items():
        print(f"{metric} : {result}")
