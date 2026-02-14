import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(df):
    df['y'] = df['y'].map({'no': 0, 'yes': 1})
    df = pd.get_dummies(df, drop_first=True)

    X = df.drop('y', axis=1)
    y = df['y']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y
