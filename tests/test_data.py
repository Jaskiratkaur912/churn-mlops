import pandas as pd
import numpy as np
import pytest
import os

def test_raw_data_exists():
    assert os.path.exists("data/raw/churn.csv"), \
        "Raw data file missing!"

def test_processed_data_exists():
    files = [
        "data/processed/X_train_fe.csv",
        "data/processed/X_test_fe.csv",
        "data/processed/y_train_fe.csv",
        "data/processed/y_test_fe.csv"
    ]
    for f in files:
        assert os.path.exists(f), f"Missing file: {f}"

def test_no_missing_values():
    X_train = pd.read_csv("data/processed/X_train_fe.csv")
    assert X_train.isnull().sum().sum() == 0, \
        "Training data has missing values!"

def test_target_is_binary():
    y_train = pd.read_csv("data/processed/y_train_fe.csv").squeeze()
    unique_values = set(y_train.unique())
    assert unique_values == {0, 1}, \
        f"Target should be 0 and 1, got {unique_values}"

def test_train_test_ratio():
    X_train = pd.read_csv("data/processed/X_train_fe.csv")
    X_test  = pd.read_csv("data/processed/X_test_fe.csv")
    total   = len(X_train) + len(X_test)
    test_ratio = len(X_test) / total
    assert 0.18 <= test_ratio <= 0.22, \
        f"Test ratio should be ~0.2, got {test_ratio:.2f}"

def test_model_exists():
    assert os.path.exists("models/best_model.pkl"), \
        "Trained model file missing!"