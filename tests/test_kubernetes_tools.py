import pytest
from unittest.mock import patch, MagicMock
from agent.tools import kubernetes_tools


def test_cluster_unavailable_graceful_handling():
    res = kubernetes_tools.list_pods(namespace="mlops")
    assert res["success"] is False
    assert res["cluster_available"] is False
    assert "error" in res


def test_unauthorized_namespace_modification_rejected():
    # operations outside approved namespaces must be rejected immediately
    res = kubernetes_tools.restart_deployment(
        deployment_name="core-api",
        namespace="kube-system",
        confirmed=True
    )
    assert res["success"] is False
    assert "Unauthorized namespace" in res["error"]


def test_state_change_requires_explicit_confirmation():
    # restarting or scaling without confirmation flag must return requires_confirmation
    restart_res = kubernetes_tools.restart_deployment(
        deployment_name="churn-inference-deployment",
        namespace="mlops",
        confirmed=False
    )
    assert restart_res["success"] is False
    assert restart_res["requires_confirmation"] is True

    scale_res = kubernetes_tools.scale_deployment(
        deployment_name="churn-inference-deployment",
        replicas=4,
        namespace="mlops",
        confirmed=False
    )
    assert scale_res["success"] is False
    assert scale_res["requires_confirmation"] is True


def test_mocked_kubernetes_pod_listing(monkeypatch):
    mock_json = """
    {
      "items": [
        {
          "metadata": {"name": "churn-pod-1", "namespace": "mlops"},
          "status": {
            "phase": "Running",
            "containerStatuses": [{"ready": true, "restartCount": 0}],
            "podIP": "10.244.0.15"
          },
          "spec": {"nodeName": "node-1"}
        }
      ]
    }
    """
    mock_res = MagicMock()
    mock_res.returncode = 0
    mock_res.stdout = mock_json

    monkeypatch.setattr(kubernetes_tools, "_run_kubectl", lambda args: mock_res)

    res = kubernetes_tools.list_pods(namespace="mlops")
    assert res["success"] is True
    assert len(res["pods"]) == 1
    assert res["pods"][0]["name"] == "churn-pod-1"
    assert res["pods"][0]["ready"] is True
