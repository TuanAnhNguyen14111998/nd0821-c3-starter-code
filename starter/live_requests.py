import requests
import json
url = "https://nd0821-c3-starter-code.herokuapp.com"

sample_payload_1 = {
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

sample_payload_2 = {
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

response = requests.post("{}/predict".format(url), data=json.dumps(sample_payload_1))
print("Response code: ", response.status_code)
print("Response from API: ",response.json())

response = requests.post("{}/predict".format(url), data=json.dumps(sample_payload_2))
print("Response code: ", response.status_code)
print("Response from API: ",response.json())
