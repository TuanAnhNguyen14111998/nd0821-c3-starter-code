import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from starter.starter.ml.model import load_model
from starter.starter.ml.model import train_model, inference
from starter.starter.ml.model import compute_model_metrics
from starter.starter.ml.data import process_data


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


def test_slice_inference(root_path):
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

    cols = ['feature_name', 'slice_len', 'instances', 'precision', 'recall', 'f1']
    df_perf = pd.DataFrame(columns=cols)

    X_train, y_train_GT, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )
    X_test, y_test_GT, encoder, lb = process_data(
        test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
    )

    test = test.reset_index(drop=True)

    model = load_model(root_path, "rfc_model.pkl")
    y_test_pred = inference(model, X_test)

    for feature in cat_features:
        df_sliced_perf = pd.DataFrame(columns=cols)
        slice_values = test[feature].value_counts().index
        for slice in slice_values:
            _idx = test.loc[test[feature] == slice].index
            predictions = y_test_pred[_idx]
            y_test = y_test_GT[_idx]
            slice_len = len(_idx)

            if slice_len < 25:
                continue
                
            precision, recall, f1_score = compute_model_metrics(y_test, predictions)
            slice_performance = pd.DataFrame(
                [[feature, slice, slice_len, precision, recall, f1_score]], 
                columns=cols)
            df_sliced_perf = df_sliced_perf.append(slice_performance, ignore_index=True)
        
        df_perf = df_perf.append(df_sliced_perf)

    # write sliced dataframe to csv to inspect later
    df_perf.to_csv(f"{root_path}/screenshots/slice_output.txt", index=False)
    assert (df_perf.f1.median() > 0.5), f"For {slice}, slice inf score"
