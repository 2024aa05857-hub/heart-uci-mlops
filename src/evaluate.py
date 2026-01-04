import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate

METRICS = ["accuracy", "precision", "recall", "roc_auc"]

def cv_scores(estimator, X, y, seed: int = 42, folds: int = 5) -> dict:
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores = cross_validate(estimator, X, y, cv=cv, scoring=METRICS, return_train_score=False)
    return {k: float(np.mean(v)) for k, v in scores.items() if k.startswith("test_")}
