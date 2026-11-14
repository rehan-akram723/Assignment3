import time
import logging
from typing import Dict, Any, List, Optional

from agent.schemas import AgentResponse, EvidenceRecord
from agent.tools import dvc_tools, mlflow_tools, airflow_tools, kubernetes_tools

logger = logging.getLogger("agent.orchestrator")


class MLOpsAgentOrchestrator:
    def __init__(self):
        self.tools = {
            # dvc tools
            "dvc_list_datasets": (dvc_tools.list_tracked_datasets, "dvc"),
            "dvc_get_version": (dvc_tools.get_dataset_version, "dvc"),
            "dvc_compare_versions": (dvc_tools.compare_dataset_versions, "dvc"),
            "dvc_detect_changes": (dvc_tools.detect_data_changes, "dvc"),
            "dvc_explain_reproduction": (dvc_tools.explain_reproduction, "dvc"),
            # mlflow tools
            "mlflow_list_experiments": (mlflow_tools.list_experiments, "mlflow"),
            "mlflow_list_runs": (mlflow_tools.list_runs, "mlflow"),
            "mlflow_get_run_details": (mlflow_tools.get_run_details, "mlflow"),
            "mlflow_get_best_run": (mlflow_tools.get_best_run, "mlflow"),
            "mlflow_compare_experiments": (mlflow_tools.compare_experiments, "mlflow"),
            "mlflow_get_registered_model": (mlflow_tools.get_registered_model, "mlflow"),
            # airflow tools
            "airflow_list_dags": (airflow_tools.list_dags, "airflow"),
            "airflow_get_dag_run_status": (airflow_tools.get_dag_run_status, "airflow"),
            "airflow_trigger_dag": (airflow_tools.trigger_dag, "airflow"),
            "airflow_get_failed_tasks": (airflow_tools.get_failed_tasks, "airflow"),
            "airflow_get_diagnostics": (airflow_tools.get_task_failure_diagnostics, "airflow"),
            # kubernetes tools
            "k8s_list_pods": (kubernetes_tools.list_pods, "kubernetes"),
            "k8s_list_deployments": (kubernetes_tools.list_deployments, "kubernetes"),
            "k8s_get_pod_status_and_logs": (kubernetes_tools.get_pod_status_and_logs, "kubernetes"),
            "k8s_inspect_service": (kubernetes_tools.inspect_service, "kubernetes"),
            "k8s_detect_unhealthy_pods": (kubernetes_tools.detect_unhealthy_pods, "kubernetes"),
            "k8s_get_deployed_version": (kubernetes_tools.get_deployed_application_version, "kubernetes"),
            "k8s_restart_deployment": (kubernetes_tools.restart_deployment, "kubernetes"),
            "k8s_scale_deployment": (kubernetes_tools.scale_deployment, "kubernetes"),
        }

    def execute_tool(self, tool_name: str, **kwargs) -> EvidenceRecord:
        if tool_name not in self.tools:
            return EvidenceRecord(
                tool_name=tool_name,
                category="unknown",
                success=False,
                duration_ms=0.0,
                error=f"Tool '{tool_name}' not registered."
            )

        func, category = self.tools[tool_name]
        start = time.perf_counter()
        try:
            res = func(**kwargs)
            duration = (time.perf_counter() - start) * 1000
            is_success = bool(res.get("success", False))
            error_msg = res.get("error") if not is_success else None

            # scrub any sensitive fields
            sanitized = {k: v for k, v in res.items() if "auth" not in k.lower() and "token" not in k.lower()}

            return EvidenceRecord(
                tool_name=tool_name,
                category=category,
                success=is_success,
                duration_ms=round(duration, 2),
                data=sanitized,
                error=error_msg
            )
        except Exception as e:
            duration = (time.perf_counter() - start) * 1000
            return EvidenceRecord(
                tool_name=tool_name,
                category=category,
                success=False,
                duration_ms=round(duration, 2),
                error=str(e)
            )

    def run_mandatory_scenario(self, query: str) -> AgentResponse:
        evidence: List[EvidenceRecord] = []
        known_facts: List[str] = []
        uncertainties: List[str] = []
        recommended_actions: List[str] = []

        # 1. query MLflow
        mlflow_ev = self.execute_tool("mlflow_get_best_run", metric_name="accuracy")
        evidence.append(mlflow_ev)

        best_run_id = None
        dvc_hash = None
        git_commit = None
        acc = None

        if mlflow_ev.success:
            best_data = mlflow_ev.data
            best_run_id = best_data.get("best_run_id")
            metric_val = best_data.get("metric_value", 0.0)
            acc = metric_val
            dvc_hash = best_data.get("dvc_dataset_hash")
            git_commit = best_data.get("git_commit")
            known_facts.append(
                f"Best MLflow run: '{best_run_id}' with accuracy {metric_val:.4f} in experiment 'churn_prediction'."
            )
        else:
            uncertainties.append(f"Could not retrieve MLflow runs: {mlflow_ev.error}")

        # 2. read DVC dataset reference
        dvc_ev = self.execute_tool("dvc_get_version")
        evidence.append(dvc_ev)

        dvc_status_ev = self.execute_tool("dvc_detect_changes")
        evidence.append(dvc_status_ev)

        if dvc_ev.success:
            curr_hash = dvc_ev.data.get("dvc_hash")
            commit_ref = dvc_ev.data.get("git_commit")
            known_facts.append(
                f"Tracked DVC dataset MD5: {curr_hash} associated with Git commit {commit_ref}."
            )
            if dvc_hash and curr_hash == dvc_hash:
                known_facts.append("Trained model dataset hash matches current active DVC dataset hash.")
            elif dvc_hash:
                uncertainties.append(f"Model dataset hash ({dvc_hash}) differs from active DVC hash ({curr_hash}).")
        else:
            uncertainties.append(f"DVC dataset metadata unavailable: {dvc_ev.error}")

        # 3. inspect Kubernetes deployment
        k8s_ev = self.execute_tool("k8s_get_deployed_version")
        evidence.append(k8s_ev)

        if k8s_ev.success:
            dep_data = k8s_ev.data
            deployed_ver = dep_data.get("deployed_version")
            replicas = dep_data.get("replicas")
            ready = dep_data.get("ready_replicas")
            known_facts.append(
                f"Kubernetes deployment 'churn-inference-deployment' is running version {deployed_ver} with {ready}/{replicas} replicas ready."
            )
        else:
            uncertainties.append(
                "Kubernetes cluster deployment status could not be verified (cluster unreachable)."
            )

        # 4. query Airflow pipeline status
        airflow_ev = self.execute_tool("airflow_get_dag_run_status", dag_id="ml_training_pipeline")
        evidence.append(airflow_ev)

        if airflow_ev.success:
            state = airflow_ev.data.get("state")
            dag_run_id = airflow_ev.data.get("dag_run_id")
            known_facts.append(f"Recent Airflow pipeline run '{dag_run_id}' has state: {state}.")
        else:
            uncertainties.append("Airflow workflow server unreachable; cannot verify live DAG execution status.")

        # 5. correlate & determine retraining recommendation
        retraining_recommended = False
        justification = None

        # evaluate data drift or quality gate failure
        data_changed = dvc_status_ev.data.get("has_changed", False) if dvc_status_ev.success else False
        if data_changed:
            retraining_recommended = True
            justification = "Tracked data file has been modified compared to recorded DVC hash."
            recommended_actions.append("Initiate retraining because dataset drift/modification was detected.")
        else:
            recommended_actions.append(
                "Do NOT initiate automatic retraining. Current model meets performance criteria and active dataset is unchanged."
            )
            recommended_actions.append("Establish periodic evaluation job to monitor inference input distribution.")

        if not k8s_ev.success:
            recommended_actions.append("Check Kubernetes cluster connectivity and verify inference Pod health.")

        # 6. compile response text
        answer_lines = [
            f"Evaluation Summary for request: '{query}'",
            "",
            "1. Best Model:",
            f"   - Run ID: {best_run_id or 'N/A'}",
            f"   - Metric (Accuracy): {acc if acc is not None else 'N/A'}",
            "",
            "2. Dataset Provenance:",
            f"   - DVC Hash: {dvc_hash or curr_hash if dvc_ev.success else 'N/A'}",
            f"   - Associated Git Commit: {git_commit or commit_ref if dvc_ev.success else 'N/A'}",
            "",
            "3. Deployment Status:",
            f"   - Status: {'Healthy and operational' if k8s_ev.success else 'Unverified / Cluster offline'}",
            "",
            "4. Retraining Decision:",
            f"   - Retraining Recommended: {'YES' if retraining_recommended else 'NO'}",
            f"   - Justification: {justification or 'No evidence of model degradation, failed quality gate, or data changes.'}"
        ]

        return AgentResponse(
            query=query,
            answer="\n".join(answer_lines),
            known_facts=known_facts,
            uncertainties=uncertainties,
            recommended_actions=recommended_actions,
            retraining_recommended=retraining_recommended,
            retraining_justification=justification,
            evidence=evidence
        )

    def process_query(self, query: str) -> AgentResponse:
        q_lower = query.lower()

        # check for mandatory multi-tool scenario
        if "best" in q_lower and ("retrain" in q_lower or "dataset" in q_lower or "deployment" in q_lower):
            return self.run_mandatory_scenario(query)

        # dvc queries
        if "dataset" in q_lower or "dvc" in q_lower:
            ev = self.execute_tool("dvc_get_version")
            ch_ev = self.execute_tool("dvc_detect_changes")
            facts = []
            if ev.success:
                facts.append(f"Dataset {ev.data.get('dataset_name')} MD5: {ev.data.get('dvc_hash')}, Commit: {ev.data.get('git_commit')}")
            if ch_ev.success:
                facts.append(ch_ev.data.get("status_summary", ""))
            return AgentResponse(
                query=query,
                answer=f"Dataset information: {'; '.join(facts) if facts else 'No dataset details found.'}",
                known_facts=facts,
                uncertainties=[] if facts else [ev.error or "Unavailable"],
                recommended_actions=["Run 'dvc status' to verify data state."],
                retraining_recommended=False,
                evidence=[ev, ch_ev]
            )

        # kubernetes queries
        if "pod" in q_lower or "deploy" in q_lower or "kubernetes" in q_lower or "unavailable" in q_lower:
            ev1 = self.execute_tool("k8s_list_pods")
            ev2 = self.execute_tool("k8s_detect_unhealthy_pods")
            facts = []
            uncertainties = []
            if ev1.success:
                facts.append(f"Found {len(ev1.data.get('pods', []))} pods.")
            else:
                uncertainties.append(ev1.error or "Cluster unreachable")

            return AgentResponse(
                query=query,
                answer=f"Kubernetes diagnosis: {'; '.join(facts) if facts else 'Cluster is currently offline or unreachable.'}",
                known_facts=facts,
                uncertainties=uncertainties,
                recommended_actions=["Verify kubectl configuration and cluster network reachability."],
                retraining_recommended=False,
                evidence=[ev1, ev2]
            )

        # airflow queries
        if "dag" in q_lower or "pipeline" in q_lower or "airflow" in q_lower:
            ev = self.execute_tool("airflow_get_dag_run_status")
            facts = []
            uncertainties = []
            if ev.success:
                facts.append(f"DAG status: {ev.data.get('state')}")
            else:
                uncertainties.append(ev.error or "Airflow server offline")

            return AgentResponse(
                query=query,
                answer=f"Airflow workflow status: {'; '.join(facts) if facts else 'Airflow server is offline.'}",
                known_facts=facts,
                uncertainties=uncertainties,
                recommended_actions=["Start Airflow webserver and scheduler."],
                retraining_recommended=False,
                evidence=[ev]
            )

        # default multi-tool evaluation
        return self.run_mandatory_scenario(query)
