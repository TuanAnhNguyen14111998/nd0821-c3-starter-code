import json
import os
from fastapi.testclient import TestClient

from starter.main import app

client = TestClient(app)

PREDICT_ENDPOINT = "http://127.0.0.1:8000/predict"

def test_welcome():
    """Test / endpoint with GET method"""
    r = client.get("/")

    response = json.loads(r.text)["welcome"]

    assert response == "😁😁😁😁 Welcome to my app!!! 😁😁😁😁"
    assert r.status_code == 200


def test_negative_pred():
    """Test /predict endpoint with POST method for negative prediction"""
    sample_payload = {
        "age": 40,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 2174,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States"
    }

    response = client.post(
        PREDICT_ENDPOINT,
        data=json.dumps(sample_payload)
    )

    prediction = response.json()["predict"]

    assert response.status_code == 200
    assert prediction == "Income <= 50k"


def test_positive_pred():
    """Test /predict endpoint with POST method for positive prediction"""
    sample_payload = {
        "age": 49,
        "workclass": "Self-emp-inc",
        "fnlgt": 191681,
        "education": "Some-college",
        "education-num": 10,
        "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 50,
        "native-country": "United-States"
    }

    response = client.post(
        PREDICT_ENDPOINT,
        data=json.dumps(sample_payload)
    )

    prediction = json.loads(response.text)["predict"]

    assert response.status_code == 200
    assert prediction == "Income > 50k"
