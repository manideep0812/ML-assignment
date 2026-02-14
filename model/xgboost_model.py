from xgboost import XGBClassifier

def build_model():
    return XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        eval_metric="logloss",
        random_state=42
    )
