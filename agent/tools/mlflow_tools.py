import os
from typing import Dict, Any, List, Optional
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException


os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"


def _get_client(tracking_uri: Optional[str] = None) -> MlflowClient:
    uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    return MlflowClient(tracking_uri=uri)


def list_experiments(tracking_uri: Optional[str] = None) -> Dict[str, Any]:
    try:
        client = _get_client(tracking_uri)
        experiments = client.search_experiments()
        result = [
            {
                "experiment_id": exp.experiment_id,
                "name": exp.name,
                "artifact_location": exp.artifact_location,
                "lifecycle_stage": exp.lifecycle_stage
            }
            for exp in experiments
        ]
        return {"success": True, "experiments": result}
    except Exception as e:
        return {"success": False, "error": f"Failed listing experiments: {str(e)}"}


def list_runs(experiment_name: str = "churn_prediction", max_results: int = 10, tracking_uri: Optional[str] = None) -> Dict[str, Any]:
    try:
        client = _get_client(tracking_uri)
        exp = client.get_experiment_by_name(experiment_name)
        if not exp:
            return {"success": False, "error": f"Experiment '{experiment_name}' not found."}

        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            max_results=max_results,
            order_by=["attribute.start_time DESC"]
        )

        run_summaries = []
        for r in runs:
            run_summaries.append({
                "run_id": r.info.run_id,
                "run_name": r.info.run_name,
                "status": r.info.status,
                "start_time": r.info.start_time,
                "metrics": dict(r.data.metrics),
                "params": dict(r.data.params),
                "tags": {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")},
                "dvc_dataset_hash": r.data.tags.get("dvc_dataset_hash"),
                "git_commit": r.data.tags.get("git_commit")
            })

        return {"success": True, "experiment_name": experiment_name, "runs": run_summaries}
    except Exception as e:
        return {"success": False, "error": f"Failed listing runs: {str(e)}"}


def get_run_details(run_id: str, tracking_uri: Optional[str] = None) -> Dict[str, Any]:
    try:
        client = _get_client(tracking_uri)
        run = client.get_run(run_id)

        artifacts = []
        try:
            art_list = client.list_artifacts(run_id)
            artifacts = [a.path for a in art_list]
        except Exception:
            pass

        return {
            "success": True,
            "run_id": run.info.run_id,
            "experiment_id": run.info.experiment_id,
            "status": run.info.status,
            "start_time": run.info.start_time,
            "metrics": dict(run.data.metrics),
            "params": dict(run.data.params),
            "tags": dict(run.data.tags),
            "artifacts": artifacts,
            "dvc_dataset_hash": run.data.tags.get("dvc_dataset_hash"),
            "git_commit": run.data.tags.get("git_commit")
        }
    except Exception as e:
        return {"success": False, "error": f"Failed retrieving run {run_id}: {str(e)}"}


def get_best_run(experiment_name: str = "churn_prediction", metric_name: str = "f1_score", mode: str = "max", tracking_uri: Optional[str] = None) -> Dict[str, Any]:
    runs_res = list_runs(experiment_name=experiment_name, max_results=20, tracking_uri=tracking_uri)
    if not runs_res.get("success"):
        return runs_res

    runs = runs_res.get("runs", [])
    if not runs:
        return {"success": False, "error": f"No runs found in experiment '{experiment_name}'."}

    valid_runs = [r for r in runs if metric_name in r["metrics"]]
    if not valid_runs:
        return {"success": False, "error": f"Metric '{metric_name}' not found in any runs."}

    reverse = (mode.lower() == "max")
    sorted_runs = sorted(valid_runs, key=lambda x: x["metrics"][metric_name], reverse=reverse)
    best = sorted_runs[0]
    metric_val = best["metrics"][metric_name]

    warnings = []
    # check model stability and balance
    acc = best["metrics"].get("accuracy", 0.0)
    prec = best["metrics"].get("precision", 0.0)
    rec = best["metrics"].get("recall", 0.0)

    if metric_val < 0.70:
        warnings.append(f"Target metric {metric_name} ({metric_val:.3f}) is below standard production threshold (0.70).")
    if abs(prec - rec) > 0.35:
        warnings.append(f"High imbalance between precision ({prec:.2f}) and recall ({rec:.2f}).")

    is_production_ready = (len(warnings) == 0)

    reasoning = (
        f"Selected run {best['run_id']} with highest {metric_name} ({metric_val:.4f}). "
        f"Accuracy is {acc:.4f}, Precision is {prec:.4f}, and Recall is {rec:.4f}. "
    )
    if not is_production_ready:
        reasoning += f"However, deployment caution is required: {'; '.join(warnings)}"
    else:
        reasoning += "Performance metrics are balanced and meet operational thresholds."

    return {
        "success": True,
        "best_run_id": best["run_id"],
        "metric_name": metric_name,
        "metric_value": metric_val,
        "metrics": best["metrics"],
        "params": best["params"],
        "dvc_dataset_hash": best.get("dvc_dataset_hash"),
        "git_commit": best.get("git_commit"),
        "is_production_ready": is_production_ready,
        "warnings": warnings,
        "reasoning": reasoning
    }


def compare_experiments(run_ids: List[str], metric_names: Optional[List[str]] = None, tracking_uri: Optional[str] = None) -> Dict[str, Any]:
    if not run_ids:
        return {"success": False, "error": "No run_ids provided for comparison."}

    metrics_to_check = metric_names or ["accuracy", "precision", "recall", "f1_score"]
    comparison = []

    for r_id in run_ids:
        res = get_run_details(r_id, tracking_uri)
        if res.get("success"):
            row = {
                "run_id": r_id,
                "status": res.get("status"),
                "params": res.get("params", {}),
                "metrics": {m: res.get("metrics", {}).get(m) for m in metrics_to_check},
                "dvc_dataset_hash": res.get("dvc_dataset_hash"),
                "git_commit": res.get("git_commit")
            }
            comparison.append(row)

    if not comparison:
        return {"success": False, "error": "Could not retrieve any of the specified runs."}

    analysis = f"Compared {len(comparison)} runs across metrics: {', '.join(metrics_to_check)}."
    return {
        "success": True,
        "compared_runs": comparison,
        "metrics_checked": metrics_to_check,
        "analysis": analysis
    }


def get_registered_model(model_name: str = "churn_model", tracking_uri: Optional[str] = None) -> Dict[str, Any]:
    try:
        client = _get_client(tracking_uri)
        model = client.get_registered_model(model_name)
        versions = []
        for mv in model.latest_versions:
            versions.append({
                "version": mv.version,
                "current_stage": mv.current_stage,
                "run_id": mv.run_id,
                "status": mv.status
            })
        return {
            "success": True,
            "model_name": model_name,
            "latest_versions": versions
        }
    except Exception as e:
        # handle unregistered or file-based tracking limitations gracefully
        return {
            "success": False,
            "error": f"Model registry lookup failed for '{model_name}': {str(e)}",
            "model_name": model_name,
            "latest_versions": []
        }
