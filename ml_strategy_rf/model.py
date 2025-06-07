# model.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def train_model_and_generate_signals(features: pd.DataFrame, returns: pd.Series, test_size=0.2, random_state=42):
    """
    Trains a Random Forest model and generates trading signals.

    Parameters:
    - features: DataFrame of engineered features
    - returns: Series of daily returns (used to create target)
    - test_size: proportion of data to use for testing
    - random_state: for reproducibility

    Returns:
    - signals: Series of trading signals (1 = long, 0 = cash)
    - model: trained Random Forest model
    """
    # Create target variable: 1 if next day's return is positive, else 0
    target = (returns.shift(-1) > 0).astype(int)
    
    # Align features and target
    features = features.loc[target.index].dropna()
    target = target.loc[features.index]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=test_size, shuffle=False)

    # Fit model
    model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    model.fit(X_train, y_train)

    # Predict on full dataset
    predictions = pd.Series(model.predict(features), index=features.index)

    # Return as trading signals
    return predictions, model
