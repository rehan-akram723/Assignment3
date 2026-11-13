import os
import sys
import argparse
import pandas as pd
import numpy as np

# ensure root is in python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn
from agent.tools.dvc_tools import get_dataset_version


def train(n_estimators: int = 100, max_depth: int = 4, test_size: float = 0.2, random_state: int = 42):
    data_path = os.path.join("data", "dataset.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    df = pd.read_csv(data_path)
    X = df[["age", "tenure", "monthly_charges", "total_charges"]]
    y = df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    dvc_info = get_dataset_version(data_path)
    dvc_hash = dvc_info.get("dvc_hash", "unknown")
    git_commit = dvc_info.get("git_commit", "unknown")

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("churn_prediction")

    with mlflow.start_run() as run:
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )
        clf.fit(X_train, y_train)

        preds = clf.predict(X_test)
        acc = float(accuracy_score(y_test, preds))
        prec = float(precision_score(y_test, preds, zero_division=0))
        rec = float(recall_score(y_test, preds, zero_division=0))
        f1 = float(f1_score(y_test, preds, zero_division=0))

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", random_state)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

        mlflow.set_tag("dvc_dataset_hash", dvc_hash)
        mlflow.set_tag("git_commit", git_commit)
        mlflow.set_tag("developer", "Rehan (23i-0500)")
        mlflow.set_tag("model_type", "RandomForestClassifier")

        os.makedirs("models", exist_ok=True)
        mlflow.sklearn.log_model(clf, artifact_path="model", serialization_format="pickle")

        run_id = run.info.run_id
        print(f"Run completed: {run_id} | F1: {f1:.4f} | Acc: {acc:.4f}")
        return run_id, f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=4)
    args = parser.parse_args()
    train(n_estimators=args.n_estimators, max_depth=args.max_depth)
