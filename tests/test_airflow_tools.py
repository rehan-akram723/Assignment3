import pytest
from unittest.mock import patch, MagicMock
import requests
from agent.tools import airflow_tools


def test_airflow_connection_failure():
    # offline or unreachable airflow should return structured error without crashing
    res = airflow_tools.list_dags()
    assert res["success"] is False
    assert "error" in res


def test_trigger_dag_unauthorized():
    # only approved DAGs should be permitted
    res = airflow_tools.trigger_dag("malicious_dag")
    assert res["success"] is False
    assert "Unauthorized DAG" in res["error"]


def test_task_failure_diagnostics_analysis():
    diag = airflow_tools.get_task_failure_diagnostics(
        dag_id="ml_training_pipeline",
        dag_run_id="run_123",
        task_id="apply_model_quality_gate"
    )
    assert diag["success"] is True
    assert diag["dag_id"] == "ml_training_pipeline"
    assert "recommended_fix" in diag


def test_mocked_dag_trigger(monkeypatch):
    mock_resp = MagicMock()
    mock_resp.status_code = 201
    mock_resp.json.return_value = {
        "dag_id": "ml_training_pipeline",
        "dag_run_id": "manual__2026-11-14T00:00:00",
        "state": "queued"
    }

    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: mock_resp)

    res = airflow_tools.trigger_dag("ml_training_pipeline")
    assert res["success"] is True
    assert res["dag_run_id"] == "manual__2026-11-14T00:00:00"
    assert res["state"] == "queued"
