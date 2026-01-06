import joblib
import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from src.data import download_raw, load_and_clean, split, COLS
from src.features import FeatureSpec, build_preprocess
from src.evaluate import cv_scores
from src.utils import save_json
from src.config import Paths, DEFAULT_SEED
from datetime import datetime, UTC

def main():
    paths = Paths()
    paths.model_dir.mkdir(parents=True, exist_ok=True)

    raw_path = download_raw(str(paths.data_raw))
    df = load_and_clean(raw_path)
    X_train, X_test, y_train, y_test = split(df, seed=DEFAULT_SEED)

    numeric = [c for c in COLS if c != "target"]
    pre = build_preprocess(FeatureSpec(numeric=numeric))

    candidates = {
        "logreg": LogisticRegression(max_iter=3000, class_weight="balanced", random_state=DEFAULT_SEED),
        "randomforest": RandomForestClassifier(
            n_estimators=400,
            random_state=DEFAULT_SEED,
            class_weight="balanced",
        ),
    }

    mlflow.set_experiment("heart-uci-risk")

    best = {"name": None, "test_roc_auc": -1.0}

    for name, model in candidates.items():
        pipe = Pipeline([("preprocess", pre), ("model", model)])

        run_name = f"{name}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tags({
                "model": name,
                "seed": DEFAULT_SEED,
                "framework": "sklearn",
            })
            mlflow.log_param("model", name)

            cv = cv_scores(pipe, X_train, y_train, seed=DEFAULT_SEED, folds=5)
            for k, v in cv.items():
                mlflow.log_metric(k, v)

            pipe.fit(X_train, y_train)
            proba = pipe.predict_proba(X_test)[:, 1]
            test_auc = float(roc_auc_score(y_test, proba))
            mlflow.log_metric("test_roc_auc", test_auc)

            mlflow.sklearn.log_model(pipe, artifact_path="model")

            if test_auc > best["test_roc_auc"]:
                best = {"name": name, "test_roc_auc": test_auc}
                joblib.dump(pipe, paths.model_path)

    save_json(paths.metadata_path, best)
    print("✅ Training complete. Best:", best)

if __name__ == "__main__":
    main()
