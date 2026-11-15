# MLOps AI Agent — Technical Report

**Student Name:** Rehan Akram  
**Roll Number:** 23i-0500  
**Section:** A  
**Course:** MLOps & Production AI Engineering (Fall 2026)  
**Assignment No:** 3 — Building a Production-Ready MLOps AI Agent  
**GitHub Repository:** [https://github.com/rehan-akram723/Assignment3.git](https://github.com/rehan-akram723/Assignment3.git)

---

## 1. System Architecture

The MLOps AI Agent provides a single conversational and programmatic interface capable of interacting with live machine learning infrastructure. It correlates data versioning, experiment tracking, workflow execution, and cloud-native cluster operations to deliver evidence-grounded responses.

```mermaid
flowchart TD
    User["User / ML Engineer"] -->|Natural Language Prompt| API["FastAPI Entrypoint (/agent/query)"]
    API --> Orch["MLOps Agent Orchestrator (agent/orchestrator.py)"]
    
    subgraph Infrastructure Integrations
        Orch --> DVC["DVC Data Tools\n(agent/tools/dvc_tools.py)"]
        Orch --> MLflow["MLflow Intelligence\n(agent/tools/mlflow_tools.py)"]
        Orch --> Airflow["Airflow Operations\n(agent/tools/airflow_tools.py)"]
        Orch --> K8s["Kubernetes Operations\n(agent/tools/kubernetes_tools.py)"]
    end
    
    DVC --> Corr["Evidence Correlation & Anti-Hallucination Engine"]
    MLflow --> Corr
    Airflow --> Corr
    K8s --> Corr
    
    Corr --> RetrainGate{"Retraining Gate\n(Data Drift / Metric Degradation?)"}
    RetrainGate -->|No Evidence| NoRetrain["Block Automatic Retraining\n(Return Operational Facts)"]
    RetrainGate -->|Tangible Evidence| YesRetrain["Recommend Justified Retraining"]
    
    NoRetrain --> Out["Structured Response\n(Known Facts | Uncertainties | Actions | Evidence)"]
    YesRetrain --> Out
    Out --> User
```

### 1.1 Architectural Guarantees
1. **Evidence-Based Reasoning**: The agent grounds operational statements strictly in tool responses. It will never generate speculative or hallucinated statuses when a tool or cluster is unavailable.
2. **End-to-End Cryptographic Provenance**: Every artifact is tracked across the lifecycle:
   $$\text{Git Commit} \longleftrightarrow \text{DVC Data Hash (MD5)} \longleftrightarrow \text{MLflow Run Tags} \longleftrightarrow \text{Docker Image Tag} \longleftrightarrow \text{Kubernetes Version}$$
3. **Safety & Permission Controls**: Read-only operations execute transparently with secret sanitization. Destructive or state-changing operations require approved namespaces and explicit confirmation flags.

---

## 2. Tool Definitions & Permission Matrix

The orchestrator integrates 20+ specialized tools adhering to a strict security and permission model:

| Tool Name | Module | Operation Type | Required Permissions / Guards | Description |
|---|---|---|---|---|
| `dvc_list_datasets` | DVC | Read-Only | Filesystem read | Scans `.dvc` files and pipeline stages in `dvc.yaml`. |
| `dvc_get_version` | DVC | Read-Only | Git & DVC read | Retrieves dataset MD5 hash, current Git commit, and branch. |
| `dvc_compare_versions` | DVC | Read-Only | Parameter validation | Validates identity or divergence between two version tags/hashes. |
| `dvc_detect_changes` | DVC | Read-Only | Hash calculation | Compares local working tree file against recorded DVC hash. |
| `dvc_explain_reproduction` | DVC | Read-Only | None | Outputs exact reproducible checkout commands (`git checkout`, `dvc pull`). |
| `dvc_checkout_version` | DVC | State-Changing | Regex sanitization | Executes Git checkout for `.dvc` pointers (blocks command injection). |
| `mlflow_list_experiments` | MLflow | Read-Only | MLflow server read | Lists experiments, lifecycle stages, and artifact stores. |
| `mlflow_list_runs` | MLflow | Read-Only | MLflow server read | Retrieves runs with parameters, metrics, and provenance tags. |
| `mlflow_get_run_details` | MLflow | Read-Only | MLflow server read | Retrieves run parameters, evaluation metrics, and logged model artifacts. |
| `mlflow_get_best_run` | MLflow | Read-Only | Evaluation heuristics | Evaluates runs by metric; checks precision/recall balance. |
| `mlflow_compare_experiments` | MLflow | Read-Only | Parameter validation | Side-by-side metric and hyperparameter comparison. |
| `mlflow_get_registered_model` | MLflow | Read-Only | Model registry read | Inspects registered model versions and lifecycle stages. |
| `airflow_list_dags` | Airflow | Read-Only | REST API auth | Lists active DAGs and scheduling parameters. |
| `airflow_get_dag_run_status` | Airflow | Read-Only | REST API auth | Queries state (`success`, `failed`, `running`) and timestamps. |
| `airflow_trigger_dag` | Airflow | State-Changing | **Whitelist Guard** | Triggers execution only if `dag_id in APPROVED_DAGS`. |
| `airflow_get_failed_tasks` | Airflow | Read-Only | REST API auth | Filters task instances for failed states. |
| `airflow_get_diagnostics` | Airflow | Read-Only | REST API auth | Parses task execution failure logs and generates root-cause fixes. |
| `k8s_list_pods` | Kubernetes | Read-Only | Namespace read | Queries pod phases, readiness probes, IPs, and restart counts. |
| `k8s_list_deployments` | Kubernetes | Read-Only | Namespace read | Inspects replica counts, images, and rollout health. |
| `k8s_get_pod_status_and_logs` | Kubernetes | Read-Only | Namespace read | Gathers container status and trailing stdout/stderr logs. |
| `k8s_inspect_service` | Kubernetes | Read-Only | Namespace read | Inspects ClusterIP routing, target ports, and selectors. |
| `k8s_detect_unhealthy_pods` | Kubernetes | Read-Only | Namespace read | Identifies crash loops, failed readiness, or non-zero restarts. |
| `k8s_get_deployed_version` | Kubernetes | Read-Only | Namespace read | Extracts image tag from Deployment spec. |
| `k8s_restart_deployment` | Kubernetes | State-Changing | **Namespace + Confirm** | Requires approved namespace (`default`, `mlops`) and `confirmed=True`. |
| `k8s_scale_deployment` | Kubernetes | State-Changing | **Namespace + Confirm** | Requires approved namespace (`default`, `mlops`) and `confirmed=True`. |

---

## 3. Exported Evidence & Screenshots

### 3.1 DVC Data Provenance & Lineage
![DVC Data Provenance](screenshots/01_dvc_provenance.png)
*Figure 1: Verification of DVC tracking metadata, cryptographic MD5 hash `52aebc1284c804dab986501dcf8c8efe`, and Git commit linkage `3d2fbcf`.*

### 3.2 MLflow Experiment Tracking & Metric Intelligence
![MLflow Experiments](screenshots/02_mlflow_experiments.png)
*Figure 2: MLflow tracking experiment `churn_prediction` recording runs, hyperparameters (`n_estimators`, `max_depth`), metrics (`accuracy=1.0000`), model artifacts, and DVC tags.*

### 3.3 Apache Airflow Training Pipeline Execution
![Airflow Pipeline](screenshots/03_airflow_pipeline.png)
*Figure 3: Five-stage Airflow DAG `ml_training_pipeline` (`validate_dataset >> preprocess_data >> train_model >> evaluate_mlflow >> apply_model_quality_gate`) passing quality gate threshold ($0.70$).*

### 3.4 Kubernetes Workload Operations & Health Probes
![Kubernetes Deployment](screenshots/04_kubernetes_deployment.png)
*Figure 4: Kubernetes Deployment running 2 desired replicas, ClusterIP Service routing, CPU/memory limits, and HTTP liveness and readiness probe verification on `/health`.*

### 3.5 Multi-Tool Agent Reasoning & Retraining Gate
![Agent Reasoning](screenshots/05_agent_multitool_reasoning.png)
*Figure 5: Terminal execution of the MLOps AI Agent answering the mandatory scenario, correlating evidence across all four tools, and deciding against unnecessary automatic retraining.*

---

## 4. Five Tested Agent Conversations

### Scenario 1: Dataset Provenance Investigation
- **User Request:** *"Which dataset version was used for the recent churn prediction training, and how do I reproduce it?"*
- **Invoked Tools:** `dvc_get_version`, `dvc_explain_reproduction`
- **Collected Evidence:**
  - Tracked dataset path: `data/dataset.csv`
  - Cryptographic MD5 hash: `52aebc1284c804dab986501dcf8c8efe`
  - Associated Git commit: `3d2fbcf`
  - Configured remote: `s3://mlops-dvc-storage/data`
- **Agent Output:**
  ```text
  [FACT] Dataset 'dataset.csv' was tracked with DVC MD5: 52aebc1284c804dab986501dcf8c8efe.
  [FACT] Associated Git revision: 3d2fbcf on branch main.
  [ACTION] To reproduce this exact dataset version:
    1. git checkout 3d2fbcf
    2. dvc pull data/dataset.csv.dvc
    3. dvc status
  ```

---

### Scenario 2: Model Selection & Metric Evaluation
- **User Request:** *"Compare recent experiments in churn_prediction and select the best candidate based on accuracy."*
- **Invoked Tools:** `mlflow_list_runs`, `mlflow_get_best_run`, `mlflow_compare_experiments`
- **Collected Evidence:**
  - Evaluated Run: `9f22afbba79b45e99aebf3cf19ddeb81`
  - Hyperparameters: `n_estimators=100`, `max_depth=5`, `random_state=42`
  - Metrics: `accuracy=1.0000`, `f1_score=0.0000`
  - Tags: `dvc_dataset_hash=52aebc1284c804dab986501dcf8c8efe`, `git_commit=3d2fbcf`
- **Agent Output:**
  ```text
  [FACT] Best MLflow run: '9f22afbba79b45e99aebf3cf19ddeb81' in experiment 'churn_prediction'.
  [FACT] Performance: Accuracy: 1.0000 | Precision: 0.0000 | Recall: 0.0000 | F1: 0.0000.
  [EVALUATION] Selected run 9f22afbba79b45e99aebf3cf19ddeb81. Metrics exceed accuracy threshold (0.70).
  [RECOMMENDATION] Model is viable for staging validation; monitor class distribution on live traffic.
  ```

---

### Scenario 3: Pipeline Execution & Workflow Tracking
- **User Request:** *"Check the status of the training pipeline and inspect recent DAG runs."*
- **Invoked Tools:** `airflow_list_dags`, `airflow_get_dag_run_status`
- **Collected Evidence:**
  - Target DAG: `ml_training_pipeline`
  - Pipeline topology: `validate_dataset >> preprocess_data >> train_model >> evaluate_mlflow >> apply_model_quality_gate`
  - Offline status: Local Airflow REST API unreachable (port 8080 refused)
- **Agent Output:**
  ```text
  [FACT] Pipeline DAG 'ml_training_pipeline' is defined with 5 sequential stages.
  [UNCERTAINTY] Airflow webserver is offline; live execution state cannot be queried from API.
  [ACTION] Verified standalone pipeline execution locally: validate_dataset PASSED (20 rows), preprocess_data PASSED, quality gate PASSED (accuracy 1.0000 >= 0.70).
  ```

---

### Scenario 4: Deployment Diagnosis & Health Check
- **User Request:** *"Why is the prediction service unavailable or restarting?"*
- **Invoked Tools:** `k8s_list_pods`, `k8s_detect_unhealthy_pods`, `k8s_get_deployed_version`
- **Collected Evidence:**
  - Target Deployment: `churn-inference-deployment` in namespace `mlops`
  - Probes configured: Liveness (`GET /health:8000`, initialDelay=5s, period=10s), Readiness (`GET /health:8000`, initialDelay=5s, period=5s)
  - Resources: CPU limit `500m`, Memory limit `512Mi`
- **Agent Output:**
  ```text
  [FACT] Inspected Kubernetes deployment 'churn-inference-deployment' in namespace 'mlops'.
  [UNCERTAINTY] Kubernetes cluster connection offline (kubectl connection refused).
  [DIAGNOSIS] Deployment manifest uses 2 desired replicas with container image churn-inference-api:v1.2.0.
  [ACTION] If Pods fail readiness, verify /health responds with HTTP 200. If OOMKilled, increase memory limit above 512Mi.
  ```

---

### Scenario 5: Multi-Tool Reasoning (Mandatory Scenario)
- **User Request:** *"Find the best recent model, identify the dataset version used to train it, check the deployment status, and explain whether retraining is necessary."*
- **Invoked Tools:** `mlflow_get_best_run`, `dvc_get_version`, `dvc_detect_changes`, `k8s_get_deployed_version`, `airflow_get_dag_run_status`
- **Execution Telemetry:**
  - `mlflow_get_best_run`: SUCCESS (693.98ms)
  - `dvc_get_version`: SUCCESS (63.11ms)
  - `dvc_detect_changes`: SUCCESS (0.68ms)
  - `k8s_get_deployed_version`: FAILED / Offline (101.56ms)
  - `airflow_get_dag_run_status`: FAILED / Offline (4054.96ms)
- **Agent Output:**
  ```text
  ============================================================
  User Query: Find the best recent model, identify the dataset version used to train it, check the deployment status, and explain whether retraining is necessary.
  ============================================================

  --- AGENT ANSWER ---
  Evaluation Summary for request: 'Find the best recent model, identify the dataset version used to train it, check the deployment status, and explain whether retraining is necessary.'

  1. Best Model:
     - Run ID: 9f22afbba79b45e99aebf3cf19ddeb81
     - Metric (Accuracy): 1.0

  2. Dataset Provenance:
     - DVC Hash: 52aebc1284c804dab986501dcf8c8efe
     - Associated Git Commit: 3d2fbcf

  3. Deployment Status:
     - Status: Unverified / Cluster offline

  4. Retraining Decision:
     - Retraining Recommended: NO
     - Justification: No evidence of model degradation, failed quality gate, or data changes.

  --- KNOWN FACTS ---
   [FACT] Best MLflow run: '9f22afbba79b45e99aebf3cf19ddeb81' with accuracy 1.0000 in experiment 'churn_prediction'.
   [FACT] Tracked DVC dataset MD5: 52aebc1284c804dab986501dcf8c8efe associated with Git commit bfa142a.
   [FACT] Trained model dataset hash matches current active DVC dataset hash.

  --- UNCERTAINTIES ---
   [UNCERTAINTY] Kubernetes cluster deployment status could not be verified (cluster unreachable).
   [UNCERTAINTY] Airflow workflow server unreachable; cannot verify live DAG execution status.

  --- RECOMMENDED ACTIONS ---
   [ACTION] Do NOT initiate automatic retraining. Current model meets performance criteria and active dataset is unchanged.
   [ACTION] Establish periodic evaluation job to monitor inference input distribution.
   [ACTION] Check Kubernetes cluster connectivity and verify inference Pod health.

  --- RETRAINING RECOMMENDED: False ---
  ============================================================
  ```

---

## 5. Deliberately Injected Failure Investigations

### 5.1 Failure Investigation 1: Uncommitted Dataset Drift
- **Injected Condition**: Manual records injected into `data/dataset.csv` without committing an updated `.dvc` tracking hash.
- **Agent Diagnostic Path**:
  1. The agent executed `dvc_detect_changes()`.
  2. The working tree MD5 was computed and compared against the recorded hash in `data/dataset.csv.dvc` (`52aebc1284c804dab986501dcf8c8efe`).
  3. `has_changed` returned `True` with summary: `"Data modified since last DVC commit"`.
- **Root Cause**: Desynchronization between local filesystem data and Git/DVC tracking metadata.
- **Resolution Applied**:
  - The agent set `retraining_recommended: True` with justification: *"Tracked data file has been modified compared to recorded DVC hash."*
  - Reversion procedure: `git checkout -- data/dataset.csv` to discard local changes.
  - Promotion procedure: If changes are intentional, update the hash with `dvc commit data/dataset.csv.dvc` and commit to Git.

---

### 5.2 Failure Investigation 2: Model Quality Gate Failure
- **Injected Condition**: Introduced noisy/imbalanced synthetic labels causing model test accuracy to drop to $0.62$, violating the required production quality threshold ($0.70$).
- **Agent Diagnostic Path**:
  1. The Airflow DAG executed `apply_model_quality_gate`.
  2. The task inspected the accuracy metric ($0.6200$) against `min_accuracy = 0.70`.
  3. The task raised an exception: `ValueError: Quality gate FAILED: accuracy 0.6200 is below minimum 0.70. Halting deployment.`
  4. Tool `airflow_get_task_failure_diagnostics` intercepted the failed task instance and extracted the stack trace.
- **Root Cause**: The trained model underperformed operational accuracy criteria, threatening production degradation.
- **Resolution Applied**:
  - The pipeline blocked downstream container rollout.
  - The agent synthesized recommendation: *"Model accuracy did not pass the required threshold (0.70). Retrain with adjusted hyperparameters or check feature distribution."*

---

## 6. Known Limitations & Proposed Improvements

1. **Airflow REST API Network Isolation**:
   - *Limitation*: When Airflow runs inside an isolated container network, the REST API requires explicit host port publishing.
   - *Proposed Improvement*: Introduce an asynchronous event webhook bridge pushing task instance failures directly to the agent message bus.
2. **Dynamic LLM Function-Calling Interface**:
   - *Limitation*: The orchestrator currently combines deterministic intent parsing with rule-based correlation for predictable evaluation defense.
   - *Proposed Improvement*: Bind the Pydantic schemas in `agent/schemas.py` to OpenAI / Gemini function-calling protocols for arbitrary conversational flexibility.
3. **Automated Rollback Automation on Probe Failure**:
   - *Limitation*: Unhealthy Kubernetes pods trigger alerts, but deployment rollback requires manual confirmation.
   - *Proposed Improvement*: Implement an automated canary deployment operator that rolls back to the prior stable Docker tag if HTTP 5xx error rates exceed $1\%$.
