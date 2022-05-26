import pytest
import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from starter.starter.ml.data import process_data


@pytest.fixture(scope='session')
def root_path():
    root_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

    return root_path


@pytest.fixture(scope='session')
def data(root_path):
    data = pd.read_csv(f'{root_path}/data/census_clean.csv')
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    train, test = train_test_split(data, test_size=0.20, random_state=42)
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    # Proces the test data with the process_data function.
    X_test, y_test, encoder, lb = process_data(
        test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
    )

    return X_train, y_train, X_test, y_test
