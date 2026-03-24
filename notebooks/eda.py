import pandas as pd
import numpy as np

df = pd.read_csv("data/raw/churn.csv")

# basic info
print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nData types:\n", df.dtypes)
print("\nMissing values:\n", df.isnull().sum())
print("\nChurn distribution:\n", df['Churn'].value_counts())
print("\nChurn %:\n", df['Churn'].value_counts(normalize=True)*100)