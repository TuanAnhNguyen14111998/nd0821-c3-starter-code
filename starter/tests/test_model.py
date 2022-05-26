import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from starter.ml.model import load_model
from starter.ml.model import train_model, inference
from starter.ml.model import compute_model_metrics

def test_load_model(root_path):
    model = load_model(root_path, "rfc_model.pkl")

    assert isinstance(model, RandomForestClassifier)


def test_train_model(data):
    X_train, y_train, X_test, y_test = data
    model = train_model(X_train, y_train)

    assert isinstance(model, RandomForestClassifier)


def test_inference(root_path, data):
    model = load_model(root_path, "rfc_model.pkl")
    X_train, y_train, X_test, y_test = data
    y_pred = inference(model, X_test)

    assert len(y_pred) == len(X_test)


def test_compute_model_metrics(root_path, data):
    model = load_model(root_path, "rfc_model.pkl")
    X_train, y_train, X_test, y_test = data
    y_pred = inference(model, X_test)

    precision, recall, fbeta = compute_model_metrics(y=y_test, preds=y_pred)

    assert precision > 0.0
    assert recall > 0.0
    assert fbeta > 0.0
