import pytest
from training.train import train
from agent.tools.mlflow_tools import list_runs


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    # seed baseline runs if experiment is not populated
    res = list_runs("churn_prediction")
    if not res.get("success") or len(res.get("runs", [])) < 2:
        train(n_estimators=50, max_depth=3)
        train(n_estimators=100, max_depth=5)
