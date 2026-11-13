import os
import sys
import logging
from datetime import datetime, timedelta
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("airflow.training_pipeline")

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "dataset.csv")


def validate_dataset(**kwargs):
    logger.info("Validating dataset integrity...")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Missing dataset at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    if df.empty:
        raise ValueError("Dataset is empty.")

    required_cols = {"age", "tenure", "monthly_charges", "total_charges", "churn"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    logger.info(f"Dataset validation passed. Total records: {len(df)}")
    return {"status": "valid", "rows": len(df)}


def preprocess_data(**kwargs):
    logger.info("Starting data preprocessing...")
    df = pd.read_csv(DATA_PATH)
    df = df.dropna()
    processed_count = len(df)
    logger.info(f"Preprocessing completed. Clean records: {processed_count}")
    return {"status": "preprocessed", "clean_rows": processed_count}


def train_model(**kwargs):
    logger.info("Executing model training...")
    from training.train import train
    run_id, f1 = train(n_estimators=50, max_depth=3)
    logger.info(f"Training completed. MLflow run: {run_id}, F1: {f1:.4f}")
    return {"run_id": run_id, "f1": f1}


def evaluate_mlflow(**kwargs):
    logger.info("Evaluating model metrics from MLflow...")
    ti = kwargs.get("ti")
    train_res = ti.xcom_pull(task_ids="train_model") if ti else None
    run_id = train_res.get("run_id") if isinstance(train_res, dict) else None

    from agent.tools.mlflow_tools import get_best_run
    best = get_best_run(metric_name="accuracy")
    logger.info(f"Best run evaluation: {best.get('best_run_id')}, Acc: {best.get('metric_value')}")
    return {"evaluated_run": run_id or best.get("best_run_id"), "metrics": best.get("metrics")}


def apply_model_quality_gate(**kwargs):
    logger.info("Evaluating model quality gate criteria...")
    ti = kwargs.get("ti")
    eval_res = ti.xcom_pull(task_ids="evaluate_mlflow") if ti else None

    # threshold check: require accuracy >= 0.70
    min_accuracy = 0.70
    acc = 0.0
    if eval_res and isinstance(eval_res, dict) and "metrics" in eval_res:
        acc = eval_res["metrics"].get("accuracy", 0.0)
    else:
        from agent.tools.mlflow_tools import get_best_run
        best = get_best_run(metric_name="accuracy")
        acc = best.get("metric_value", 0.0)

    logger.info(f"Quality gate check: accuracy={acc:.4f}, required={min_accuracy}")
    if acc < min_accuracy:
        raise ValueError(f"Quality gate FAILED: accuracy {acc:.4f} is below minimum {min_accuracy}. Halting deployment.")

    logger.info("Quality gate PASSED. Model eligible for deployment promotion.")
    return {"gate_passed": True, "accuracy": acc}


# airflow DAG definition with fallback shim
default_args = {
    "owner": "Rehan",
    "depends_on_past": False,
    "start_date": datetime(2026, 1, 1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(seconds=15),
}

try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator

    dag = DAG(
        "ml_training_pipeline",
        default_args=default_args,
        description="ML training pipeline with quality gating",
        schedule_interval=None,
        catchup=False,
    )

    t1 = PythonOperator(task_id="validate_dataset", python_callable=validate_dataset, dag=dag)
    t2 = PythonOperator(task_id="preprocess_data", python_callable=preprocess_data, dag=dag)
    t3 = PythonOperator(task_id="train_model", python_callable=train_model, dag=dag)
    t4 = PythonOperator(task_id="evaluate_mlflow", python_callable=evaluate_mlflow, dag=dag)
    t5 = PythonOperator(task_id="apply_model_quality_gate", python_callable=apply_model_quality_gate, dag=dag)

    t1 >> t2 >> t3 >> t4 >> t5

except ImportError:
    dag = None
