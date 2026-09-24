import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_and_preprocess_data():
    """Load and preprocess ransomware/benign datasets"""
    try:
        # Load datasets
        benign = pd.read_csv("data/benign.csv")
        ransom = pd.read_csv("data/ransom.csv")

        # Add labels
        benign['Label'] = 0
        ransom['Label'] = 1

        # Combine datasets
        df = pd.concat([benign, ransom], ignore_index=True)

        # Remove non-feature columns (KEEP 'Label' for now)
        cols_to_remove = ['Flow ID', 'Source IP', 'Destination IP', 'Timestamp']
        df = df.drop(columns=[col for col in cols_to_remove if col in df.columns])

        # Clean data
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)

        # Separate features and labels (NOW 'Label' exists)
        y = df['Label']
        X = df.drop('Label', axis=1)

        # Keep numeric columns only
        X = X.select_dtypes(include=['number'])

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        # Scaling
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        return X_train, X_test, y_train, y_test, scaler, X.columns.tolist()

    except Exception as e:
        raise RuntimeError(f"Data loading/preprocessing failed: {str(e)}")