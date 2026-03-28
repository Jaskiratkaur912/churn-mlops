import pytest
import pandas as pd
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_ingestion import validate_data, split_data

def test_validate_data_passes():
    # create a dummy dataframe with required columns
    df = pd.DataFrame({
        'customerID':     ['1', '2'],
        'tenure':         [12, 24],
        'MonthlyCharges': [65.4, 80.2],
        'TotalCharges':   ['784.8', '1924.8'],
        'Churn':          [0, 1]
    })
    result = validate_data(df)
    assert result == True

def test_validate_data_fails_missing_column():
    # dataframe missing required column
    df = pd.DataFrame({
        'customerID': ['1', '2'],
        'tenure':     [12, 24]
    })
    with pytest.raises(ValueError):
        validate_data(df)

def test_split_data_shapes():
    df = pd.DataFrame({
        'customerID':     ['1', '2', '3', '4', '5',
                           '6', '7', '8', '9', '10'],
        'tenure':         [12, 24, 6, 36, 48,
                           10, 20, 30, 40, 50],
        'MonthlyCharges': [65, 80, 55, 90, 70,
                           60, 75, 85, 95, 50],
        'TotalCharges':   ['780', '1920', '330', '3240', '3360',
                           '600', '1500', '2550', '3800', '2500'],
        'Churn':          [0, 1, 0, 1, 0,
                           0, 1, 0, 1, 0]
    })
    X_train, X_test, y_train, y_test = split_data(df, 'Churn', test_size=0.2)
    assert len(X_train) == 8
    assert len(X_test)  == 2

def test_no_data_leakage():
    # make sure train and test have no overlapping indices
    df = pd.DataFrame({
        'customerID':     [str(i) for i in range(100)],
        'tenure':         list(range(100)),
        'MonthlyCharges': [float(i) for i in range(100)],
        'TotalCharges':   [str(float(i*10)) for i in range(100)],
        'Churn':          [i % 2 for i in range(100)]
    })
    X_train, X_test, y_train, y_test = split_data(df, 'Churn', test_size=0.2)
    train_idx = set(X_train.index)
    test_idx  = set(X_test.index)
    assert len(train_idx.intersection(test_idx)) == 0