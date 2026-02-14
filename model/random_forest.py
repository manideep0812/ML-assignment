from sklearn.ensemble import RandomForestClassifier

def build_model():
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        random_state=42
    )
