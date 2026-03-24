import os
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import joblib

def load_processed_data():
    X_train = pd.read_csv("data/processed/X_train.csv")
    X_test  = pd.read_csv("data/processed/X_test.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
    y_test  = pd.read_csv("data/processed/y_test.csv").squeeze()
    print("Data loaded successfully!")
    return X_train, X_test, y_train, y_test

def drop_useless_columns(df: pd.DataFrame) -> pd.DataFrame:
    # customerID is just an ID — no predictive value
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])
    return df

def fix_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    # TotalCharges is stored as string — convert to number
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    return df

def encode_binary_columns(df: pd.DataFrame) -> pd.DataFrame:
    # columns that have only Yes/No → convert to 1/0
    binary_cols = [
        'Partner', 'Dependents', 'PhoneService',
        'PaperlessBilling', 'MultipleLines',
        'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport',
        'StreamingTV', 'StreamingMovies'
    ]
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map({'Yes': 1, 'No': 0, 'No phone service': 0, 'No internet service': 0})
    return df

def encode_categorical_columns(df: pd.DataFrame) -> pd.DataFrame:
    # columns with more than 2 categories → one hot encode
    cat_cols = ['gender', 'InternetService', 'Contract', 'PaymentMethod']
    for col in cat_cols:
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = pd.concat([df, dummies], axis=1)
            df = df.drop(columns=[col])
    return df

def scale_numeric_columns(df_train: pd.DataFrame, df_test: pd.DataFrame):
    # scale numeric columns — fit on train only, transform both
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    num_cols = [c for c in num_cols if c in df_train.columns]

    scaler = StandardScaler()
    df_train[num_cols] = scaler.fit_transform(df_train[num_cols])
    df_test[num_cols]  = scaler.transform(df_test[num_cols])

    # save scaler for use at prediction time
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.pkl")
    print("Scaler saved to models/scaler.pkl")

    return df_train, df_test

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    # fill missing numeric values with median
    for col in df.select_dtypes(include=np.number).columns:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"Filled missing values in {col} with median {median_val}")
    return df

def align_columns(df_train: pd.DataFrame, df_test: pd.DataFrame):
    # make sure train and test have same columns after encoding
    missing_in_test = set(df_train.columns) - set(df_test.columns)
    for col in missing_in_test:
        df_test[col] = 0
    df_test = df_test[df_train.columns]
    return df_train, df_test

def save_features(X_train, X_test, y_train, y_test):
    os.makedirs("data/processed", exist_ok=True)
    X_train.to_csv("data/processed/X_train_fe.csv", index=False)
    X_test.to_csv("data/processed/X_test_fe.csv",   index=False)
    y_train.to_csv("data/processed/y_train_fe.csv", index=False)
    y_test.to_csv("data/processed/y_test_fe.csv",   index=False)
    print("Feature engineered data saved!")
    print("X_train shape:", X_train.shape)
    print("X_test shape: ", X_test.shape)

if __name__ == "__main__":
    # load
    X_train, X_test, y_train, y_test = load_processed_data()

    # clean
    X_train = drop_useless_columns(X_train)
    X_test  = drop_useless_columns(X_test)

    X_train = fix_total_charges(X_train)
    X_test  = fix_total_charges(X_test)

    X_train = encode_binary_columns(X_train)
    X_test  = encode_binary_columns(X_test)

    X_train = encode_categorical_columns(X_train)
    X_test  = encode_categorical_columns(X_test)

    X_train = handle_missing_values(X_train)
    X_test  = handle_missing_values(X_test)

    X_train, X_test = align_columns(X_train, X_test)
    X_train, X_test = scale_numeric_columns(X_train, X_test)

    # save
    save_features(X_train, X_test, y_train, y_test)