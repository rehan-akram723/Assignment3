import pytest
from unittest.mock import MagicMock
from agent.orchestrator import MLOpsAgentOrchestrator


def test_tool_selection_and_execution():
    orchestrator = MLOpsAgentOrchestrator()
    record = orchestrator.execute_tool("dvc_get_version")
    assert record.success is True
    assert record.category == "dvc"
    assert record.duration_ms >= 0
    assert "dvc_hash" in record.data


def test_agent_does_not_invent_operational_results_when_tools_unavailable():
    # External cluster and airflow are unavailable in this environment
    orchestrator = MLOpsAgentOrchestrator()
    query = "Find the best recent model, identify the dataset version used to train it, check the deployment status, and explain whether retraining is necessary."
    response = orchestrator.process_query(query)

    # Agent must NOT invent fake pod numbers or successful states in known_facts
    for fact in response.known_facts:
        assert "cluster deployment status" not in fact.lower()

    # Agent must explicitly isolate unavailable services in uncertainties
    assert len(response.uncertainties) > 0
    has_cluster_uncertainty = any("kubernetes" in u.lower() or "cluster" in u.lower() for u in response.uncertainties)
    assert has_cluster_uncertainty is True


def test_retraining_decision_gate():
    orchestrator = MLOpsAgentOrchestrator()
    query = "Find the best recent model, identify the dataset version used to train it, check the deployment status, and explain whether retraining is necessary."
    response = orchestrator.process_query(query)

    # When dataset has not drifted and metrics meet standards, automatic retraining must be False
    assert response.retraining_recommended is False
    assert any("do not" in a.lower() for a in response.recommended_actions)


def test_structured_logging_and_sanitization():
    orchestrator = MLOpsAgentOrchestrator()
    record = orchestrator.execute_tool("mlflow_get_best_run", metric_name="accuracy")
    assert record.success is True
    assert record.tool_name == "mlflow_get_best_run"
    # verify no sensitive secrets leaked in evidence data
    for k in record.data.keys():
        assert "password" not in k.lower()
        assert "token" not in k.lower()
