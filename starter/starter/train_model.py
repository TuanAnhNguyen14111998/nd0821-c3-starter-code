# Script to train machine learning model.
# Add the necessary imports for the starter code.
import pandas as pd
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics
import pickle

# Add code to load in the data.
data = pd.read_csv('../data/census_clean.csv')

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20, random_state=42)

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
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
)

# Train and save a model
rfc_model = train_model(X_train=X_train, y_train=y_train)

# Get Predictions and model metrics 
preds = inference(model=rfc_model, X=X_test)
precision, recall, fbeta = compute_model_metrics(y=y_test, preds=preds)
print("Precision: ", precision)
print("Recall: ", recall)
print("FBeta: ", fbeta)

with open('../model/rfc_model.pkl', 'wb') as pickle_file:
    pickle.dump(rfc_model, pickle_file)

with open('../model/encoder.pkl', 'wb') as pickle_file:
    pickle.dump(encoder, pickle_file)

with open('../model/lb.pkl', 'wb') as pickle_file:
    pickle.dump(lb, pickle_file)

with open('../model/metrics.txt', "w") as metrics_file:
    metrics_file.write(f"Precision: {precision}\n")
    metrics_file.write(f"Recall: {recall}\n")
    metrics_file.write(f"FBeta: {fbeta}\n")
