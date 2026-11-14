import pytest
from agent.tools import mlflow_tools


def test_list_experiments():
    res = mlflow_tools.list_experiments()
    assert res["success"] is True
    exp_names = [e["name"] for e in res["experiments"]]
    assert "churn_prediction" in exp_names


def test_list_and_get_best_run():
    runs_res = mlflow_tools.list_runs("churn_prediction")
    assert runs_res["success"] is True
    assert len(runs_res["runs"]) > 0

    best_res = mlflow_tools.get_best_run("churn_prediction", metric_name="accuracy")
    assert best_res["success"] is True
    assert "best_run_id" in best_res
    assert "reasoning" in best_res
    assert best_res["dvc_dataset_hash"] is not None


def test_compare_experiments():
    runs_res = mlflow_tools.list_runs("churn_prediction", max_results=2)
    run_ids = [r["run_id"] for r in runs_res["runs"][:2]]
    comp = mlflow_tools.compare_experiments(run_ids)
    assert comp["success"] is True
    assert len(comp["compared_runs"]) == len(run_ids)
