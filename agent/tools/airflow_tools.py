import os
import requests
from requests.auth import HTTPBasicAuth
from typing import Dict, Any, List, Optional

APPROVED_DAGS = {"ml_training_pipeline"}


def _get_config():
    base_url = os.getenv("AIRFLOW_BASE_URL", "http://localhost:8080/api/v1").rstrip("/")
    user = os.getenv("AIRFLOW_USERNAME", "airflow")
    password = os.getenv("AIRFLOW_PASSWORD", "airflow")
    return base_url, HTTPBasicAuth(user, password)


def list_dags() -> Dict[str, Any]:
    base_url, auth = _get_config()
    try:
        resp = requests.get(f"{base_url}/dags", auth=auth, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            dags = [
                {
                    "dag_id": d.get("dag_id"),
                    "description": d.get("description"),
                    "is_paused": d.get("is_paused"),
                    "is_active": d.get("is_active")
                }
                for d in data.get("dags", [])
            ]
            return {"success": True, "dags": dags}
        return {"success": False, "error": f"Airflow API responded with HTTP {resp.status_code}: {resp.text}"}
    except requests.exceptions.RequestException as e:
        return {"success": False, "error": f"Airflow connection failed: {str(e)}"}


def get_dag_run_status(dag_id: str = "ml_training_pipeline", dag_run_id: Optional[str] = None) -> Dict[str, Any]:
    base_url, auth = _get_config()
    try:
        if dag_run_id:
            url = f"{base_url}/dags/{dag_id}/dagRuns/{dag_run_id}"
        else:
            url = f"{base_url}/dags/{dag_id}/dagRuns?order_by=-execution_date&limit=1"

        resp = requests.get(url, auth=auth, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            if not dag_run_id:
                runs = data.get("dag_runs", [])
                if not runs:
                    return {"success": False, "error": f"No DAG runs found for {dag_id}."}
                run_data = runs[0]
            else:
                run_data = data

            return {
                "success": True,
                "dag_id": dag_id,
                "dag_run_id": run_data.get("dag_run_id"),
                "state": run_data.get("state"),
                "execution_date": run_data.get("execution_date"),
                "start_date": run_data.get("start_date"),
                "end_date": run_data.get("end_date")
            }
        return {"success": False, "error": f"Airflow returned status {resp.status_code}: {resp.text}"}
    except requests.exceptions.RequestException as e:
        return {"success": False, "error": f"Failed reaching Airflow: {str(e)}"}


def trigger_dag(dag_id: str = "ml_training_pipeline", conf: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if dag_id not in APPROVED_DAGS:
        return {
            "success": False,
            "error": f"Unauthorized DAG: '{dag_id}'. Only approved DAGs ({list(APPROVED_DAGS)}) can be triggered."
        }

    base_url, auth = _get_config()
    try:
        url = f"{base_url}/dags/{dag_id}/dagRuns"
        payload = {"conf": conf or {}}
        resp = requests.post(url, json=payload, auth=auth, timeout=5)
        if resp.status_code in (200, 201):
            data = resp.json()
            return {
                "success": True,
                "dag_id": dag_id,
                "dag_run_id": data.get("dag_run_id"),
                "state": data.get("state"),
                "message": f"DAG '{dag_id}' triggered successfully."
            }
        return {"success": False, "error": f"Airflow trigger failed (HTTP {resp.status_code}): {resp.text}"}
    except requests.exceptions.RequestException as e:
        return {"success": False, "error": f"Failed reaching Airflow server: {str(e)}"}


def get_failed_tasks(dag_id: str = "ml_training_pipeline", dag_run_id: str = "") -> Dict[str, Any]:
    if not dag_run_id:
        return {"success": False, "error": "dag_run_id is required."}

    base_url, auth = _get_config()
    try:
        url = f"{base_url}/dags/{dag_id}/dagRuns/{dag_run_id}/taskInstances"
        resp = requests.get(url, auth=auth, timeout=5)
        if resp.status_code == 200:
            tasks = resp.json().get("task_instances", [])
            failed = [
                {
                    "task_id": t.get("task_id"),
                    "state": t.get("state"),
                    "try_number": t.get("try_number"),
                    "start_date": t.get("start_date"),
                    "end_date": t.get("end_date")
                }
                for t in tasks if t.get("state") in ("failed", "upstream_failed")
            ]
            return {"success": True, "dag_id": dag_id, "dag_run_id": dag_run_id, "failed_tasks": failed}
        return {"success": False, "error": f"Airflow task check failed (HTTP {resp.status_code}): {resp.text}"}
    except requests.exceptions.RequestException as e:
        return {"success": False, "error": f"Connection error: {str(e)}"}


def get_task_failure_diagnostics(dag_id: str, dag_run_id: str, task_id: str) -> Dict[str, Any]:
    base_url, auth = _get_config()
    try:
        url = f"{base_url}/dags/{dag_id}/dagRuns/{dag_run_id}/taskInstances/{task_id}/logs/1"
        resp = requests.get(url, auth=auth, timeout=5)
        log_content = resp.text if resp.status_code == 200 else "Log retrieval unavailable."
    except Exception as e:
        log_content = f"Could not fetch raw logs: {str(e)}"

    # diagnosis rules
    recommendation = "Inspect task execution environment and retry."
    if "Quality gate FAILED" in log_content:
        recommendation = "Model accuracy did not pass the required threshold (0.70). Retrain with better hyperparameters or check feature distribution."
    elif "FileNotFoundError" in log_content or "dataset.csv" in log_content:
        recommendation = "Dataset file missing. Run DVC pull to restore the tracked dataset."
    elif "Missing required columns" in log_content:
        recommendation = "Input schema mismatch. Verify columns in data/dataset.csv."

    return {
        "success": True,
        "dag_id": dag_id,
        "dag_run_id": dag_run_id,
        "task_id": task_id,
        "error_message": f"Task '{task_id}' failed during pipeline execution.",
        "logs_snippet": log_content[:500],
        "recommended_fix": recommendation
    }
