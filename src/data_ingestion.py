import os
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def validate_data(df: pd.DataFrame) -> bool:
    required_columns = [
        'customerID', 'tenure', 'MonthlyCharges',
        'TotalCharges', 'Churn'
    ]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")
    if df.empty:
        raise ValueError("Dataframe is empty!")
    print("Data validation passed!")
    return True

def split_data(df: pd.DataFrame, target: str, test_size: float = 0.2):
    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=42,
        stratify=y
    )
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test

def save_splits(X_train, X_test, y_train, y_test, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    X_train.to_csv(output_dir + "/X_train.csv", index=False)
    X_test.to_csv(output_dir + "/X_test.csv", index=False)
    y_train.to_csv(output_dir + "/y_train.csv", index=False)
    y_test.to_csv(output_dir + "/y_test.csv", index=False)
    print("Splits saved to " + output_dir)

if __name__ == "__main__":
    df = load_data("data/raw/churn.csv")
    validate_data(df)
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    X_train, X_test, y_train, y_test = split_data(df, target='Churn')
    save_splits(X_train, X_test, y_train, y_test, "data/processed")