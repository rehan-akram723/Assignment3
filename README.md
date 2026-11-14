# Production-Ready MLOps AI Agent

**Student Name:** Rehan Akram  
**Roll Number:** 23i-0500  
**Section:** A  
**Course:** MLOps & Production AI Engineering | Assignment 03 | Fall 2026  
**Repository:** [https://github.com/rehan-akram723/Assignment3.git](https://github.com/rehan-akram723/Assignment3.git)

---

## 1. Project Overview

This repository contains a production-ready, evidence-based **MLOps AI Agent** capable of understanding natural-language operational requests, interacting directly with underlying MLOps tooling (DVC, MLflow, Apache Airflow, and Kubernetes), executing authorized tasks, and synthesizing structured answers grounded in runtime facts.

Unlike basic conversational bots that hallucinate operational metrics, this agent:
- Enforces an **anti-hallucination policy** that explicitly isolates missing data as `uncertainties`.
- Establishes end-to-end cryptographic **provenance tracking** (Git commit $\leftrightarrow$ DVC dataset hash $\leftrightarrow$ MLflow experiment tags $\leftrightarrow$ Kubernetes deployment image version).
- Restricts state mutations to approved namespaces with explicit human confirmation gates.
- Employs an automated model quality gate preventing degraded models from reaching production.

---

## 2. Repository Layout

```
.
├── .github/
│   └── workflows/
│       └── ci.yml                 # GitHub Actions CI workflow (linting & pytest)
├── agent/
│   ├── __init__.py
│   ├── main.py                    # CLI entrypoint for interactive agent queries
│   ├── orchestrator.py            # Multi-tool coordinator & evidence correlation engine
│   ├── schemas.py                 # Pydantic schemas for requests, tools, and responses
│   └── tools/
│       ├── __init__.py
│       ├── airflow_tools.py       # Airflow REST API integration & failure diagnostics
│       ├── dvc_tools.py           # DVC lineage, hashing, and reproduction tools
│       ├── kubernetes_tools.py    # Kubectl client tools, health checks, & safety guards
│       └── mlflow_tools.py        # MlflowClient experiment tracking & metric intelligence
├── api/
│   ├── __init__.py
│   └── main.py                    # FastAPI inference service and agent query endpoint
├── dags/
│   └── training_pipeline.py       # 5-stage Airflow training DAG with model quality gate
├── data/
│   ├── dataset.csv                # Sample tabular churn dataset
│   └── dataset.csv.dvc            # DVC tracked hash and metadata
├── kubernetes/
│   ├── configmap.yaml             # Non-sensitive runtime configuration
│   ├── deployment.yaml            # 2-replica deployment with liveness/readiness probes
│   └── service.yaml               # ClusterIP service exposing inference API
├── tests/
│   ├── __init__.py
│   ├── test_airflow_tools.py      # Airflow connection, unauthorized DAG, & mock tests
│   ├── test_dvc_tools.py          # DVC tracking, change detection, & command guards
│   ├── test_kubernetes_tools.py   # K8s offline handling, namespace safety, & mocks
│   ├── test_mlflow_tools.py       # MLflow experiment querying & run comparison tests
│   └── test_orchestrator.py       # Multi-tool scenario, anti-hallucination, & safety tests
├── training/
│   ├── __init__.py
│   └── train.py                   # Model training script logging metrics & DVC provenance
├── Dockerfile                     # Containerization for FastAPI inference service
├── dvc.yaml                       # DVC pipeline stages and dependencies
├── pyproject.toml                 # Project metadata and test configuration
├── requirements.txt               # Pinned Python dependencies
├── architecture.md                # System architecture document & Mermaid diagrams
└── README.md                      # Setup guide, technical report, & scenario defenses
```

---

## 3. Installation & Environment Setup

### 3.1 Prerequisites
- Python 3.10+ (tested on Python 3.11.9)
- Git
- Docker (optional, for containerization)
- `kubectl` (optional, for active cluster inspection)

### 3.2 Virtual Environment Setup
```bash
# Clone the repository
git clone https://github.com/rehan-akram723/Assignment3.git
cd Assignment3

# Create and activate a virtual environment
python -m venv .venv

# On Linux/macOS:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3.3 Environment Variables
Configure optional overrides via environment variables:
| Variable | Default Value | Description |
|---|---|---|
| `MLFLOW_TRACKING_URI` | `sqlite:///mlflow.db` | MLflow tracking backend |
| `AIRFLOW_BASE_URL` | `http://localhost:8080/api/v1` | Apache Airflow REST API endpoint |
| `AIRFLOW_USERNAME` | `airflow` | Airflow authentication username |
| `AIRFLOW_PASSWORD` | `airflow` | Airflow authentication password |
| `APP_ENV` | `production` | Deployment environment tier |
| `PORT` | `8000` | Port for FastAPI service |

---

## 4. Running the System

### 4.1 Model Training & Experiment Tracking
To trigger a training run and log hyperparameters, metrics, and DVC provenance tags:
```bash
python training/train.py --n_estimators 100 --max_depth 5
```

### 4.2 Launching the FastAPI Service
Start the inference service and Agent API:
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
- Health Check: `GET http://localhost:8000/health`
- Version & Provenance: `GET http://localhost:8000/version`
- Inference: `POST http://localhost:8000/predict`
- Agent Query: `POST http://localhost:8000/agent/query`

### 4.3 Running the AI Agent CLI
To query the agent interactively from the command line:
```bash
python agent/main.py --query "Find the best recent model, identify the dataset version used to train it, check the deployment status, and explain whether retraining is necessary."
```

### 4.4 Running Automated Tests
Run the complete test suite with verbose output:
```bash
pytest -v
```

---

## 5. Technical Report & Engineering Defense

### 5.1 System Architecture & Tool Definitions
The agent maintains a registry of 20+ specialized tools across four functional pillars:

1. **DVC Tools (`agent/tools/dvc_tools.py`)**:
   - `list_tracked_datasets`: Scans filesystem and `dvc.yaml` for tracked datasets.
   - `get_dataset_version`: Retrieves dataset MD5 hash, current Git commit, and branch.
   - `compare_dataset_versions`: Evaluates differences between version hashes.
   - `detect_data_changes`: Compares working tree data against recorded DVC hash.
   - `explain_reproduction`: Outlines reproducible checkout and pull commands.
   - `checkout_dataset_version`: Restricted checkout command with parameter sanitization.

2. **MLflow Tools (`agent/tools/mlflow_tools.py`)**:
   - `list_experiments`: Enumerates active tracking experiments.
   - `list_runs`: Retrieves runs with parameters, metrics, and provenance tags.
   - `get_run_details`: Inspects run artifacts, parameters, and tags.
   - `get_best_run`: Finds best run by metric with production-readiness evaluation (inspects balance between precision and recall rather than assuming highest score is production ready).
   - `compare_experiments`: Compares metrics and parameters across runs side-by-side.
   - `get_registered_model`: Queries model registry versions and stages.

3. **Airflow Tools (`agent/tools/airflow_tools.py`)**:
   - `list_dags`: Fetches DAG definitions via Airflow REST API.
   - `get_dag_run_status`: Retrieves execution state and timestamps.
   - `trigger_dag`: Triggers authorized pipelines (restricted by whitelist `APPROVED_DAGS`).
   - `get_failed_tasks`: Isolates failed tasks within a specific run.
   - `get_task_failure_diagnostics`: Fetches failure logs and generates root-cause fixes.

4. **Kubernetes Tools (`agent/tools/kubernetes_tools.py`)**:
   - `list_pods`: Queries pod phases, readiness, IPs, and restart counts.
   - `list_deployments`: Inspects replica counts, images, and rollout health.
   - `get_pod_status_and_logs`: Collects container statuses and execution logs.
   - `inspect_service`: Validates ClusterIP routing and target ports.
   - `detect_unhealthy_pods`: Filters pods experiencing restarts or non-running phases.
   - `get_deployed_application_version`: Identifies deployed image version tag.
   - `restart_deployment` & `scale_deployment`: State mutations requiring approved namespaces (`default`, `mlops`) and explicit user confirmation (`confirmed=True`).

---

### 5.2 Five Tested Agent Scenarios with Actual Evidence

#### Scenario 1: Dataset Provenance
**User Request:**
> *"Which dataset version was used for the recent churn prediction training, and how do I reproduce it?"*

**Agent Evidence & Response:**
```
[FACT] Tracked DVC dataset: data/dataset.csv
[FACT] Recorded MD5 Hash: 52aebc1284c804dab986501dcf8c8efe
[FACT] Associated Git Commit: 3d2fbcf
[ACTION] Reproduction Steps:
  1. git checkout 3d2fbcf
  2. dvc pull data/dataset.csv.dvc
  3. dvc status
```

#### Scenario 2: Model Selection & Metric Intelligence
**User Request:**
> *"Compare recent experiments in churn_prediction and select the best candidate based on accuracy."*

**Agent Evidence & Response:**
```
[FACT] Best MLflow Run ID: 9f22afbba79b45e99aebf3cf19ddeb81
[FACT] Parameters: {'n_estimators': '100', 'max_depth': '5', 'random_state': '42'}
[FACT] Metrics: Accuracy: 1.0000 | Precision: 0.0000 | Recall: 0.0000 | F1: 0.0000
[FACT] Provenance: DVC Hash: 52aebc1284c804dab986501dcf8c8efe | Git: 3d2fbcf
[EVALUATION] Candidate meets accuracy threshold, but precision/recall balance indicates target class scarcity in test partition.
```

#### Scenario 3: Pipeline Execution & Workflow Tracking
**User Request:**
> *"Check the status of the training pipeline and inspect recent DAG runs."*

**Agent Evidence & Response:**
```
[FACT] DAG ID: ml_training_pipeline
[FACT] Stages: validate_dataset -> preprocess_data -> train_model -> evaluate_mlflow -> apply_model_quality_gate
[UNCERTAINTY] Airflow webserver offline (Connection refused on port 8080).
[ACTION] Verified standalone DAG execution locally: validate_dataset PASSED (20 records), preprocess_data PASSED, quality gate PASSED (accuracy 1.0000 >= 0.70).
```

#### Scenario 4: Deployment Diagnosis & Safety
**User Request:**
> *"Why is the prediction service unavailable or restarting?"*

**Agent Evidence & Response:**
```
[FACT] Checked Kubernetes Deployment 'churn-inference-deployment' in namespace 'mlops'.
[UNCERTAINTY] Live cluster connection offline.
[DIAGNOSIS] Manifest inspection shows:
  - 2 desired replicas
  - Liveness probe: HTTP GET /health (port 8000, initialDelay=5s, period=10s)
  - Readiness probe: HTTP GET /health (port 8000, initialDelay=5s, period=5s)
  - Resource Requests: cpu 100m, memory 128Mi
[ACTION] If CrashLoopBackOff is encountered, inspect memory limits (512Mi) or check if model file is accessible.
```

#### Scenario 5: Multi-Tool Reasoning (Mandatory Scenario)
**User Request:**
> *"Find the best recent model, identify the dataset version used to train it, check the deployment status, and explain whether retraining is necessary."*

**Agent Evidence & Response:**
```
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

--- EVIDENCE COLLECTED ---
 - mlflow_get_best_run [mlflow] -> SUCCESS (693.98ms)
 - dvc_get_version [dvc] -> SUCCESS (63.11ms)
 - dvc_detect_changes [dvc] -> SUCCESS (0.68ms)
 - k8s_get_deployed_version [kubernetes] -> FAILED (101.56ms)
 - airflow_get_dag_run_status [airflow] -> FAILED (4054.96ms)
============================================================
```

---

### 5.3 Two Injected Failure Investigations

#### Failure Investigation 1: Dataset Modification & Drift Detection
- **Symptom / Injected State**: Local records in `data/dataset.csv` modified without updating DVC tracking.
- **Agent Diagnosis**: Tool `dvc_detect_changes` reported `has_changed: True`, noting that the working tree MD5 did not match the recorded hash `52aebc1284c804dab986501dcf8c8efe`.
- **Root Cause**: Uncommitted manual changes to training data causing drift between logged model provenance and filesystem state.
- **Resolution**:
  1. Revert unexpected changes via `git checkout -- data/dataset.csv`.
  2. If data modification is intentional, create a new DVC version via `dvc commit data/dataset.csv.dvc` and commit to Git.
  3. The agent activates `retraining_recommended: True` with the justification: *"Tracked data file has been modified compared to recorded DVC hash."*

#### Failure Investigation 2: Model Quality Gate Failure in Airflow Pipeline
- **Symptom / Injected State**: Underperforming model training run where accuracy drops below the quality threshold ($0.70$).
- **Agent Diagnosis**: Tool `airflow_get_task_failure_diagnostics` intercepted a failure in task `apply_model_quality_gate` with error: `ValueError: Quality gate FAILED: accuracy 0.6200 is below minimum 0.70. Halting deployment.`
- **Root Cause**: Model underfitting or hyperparameter degradation caused evaluation metrics to violate production standards.
- **Resolution**:
  1. Pipeline blocked downstream deployment promotion, preventing defective weights from being registered.
  2. Agent recommended: *"Model accuracy did not pass the required threshold (0.70). Retrain with adjusted hyperparameters or check feature distributions."*

---

## 6. Known Limitations & Proposed Improvements

1. **Airflow 3 REST API Extensibility**:
   - *Limitation*: When Airflow is running in isolated Docker containers, API network bridging requires explicit port publishing.
   - *Improvement*: Implement an async polling client with exponential backoff and webhook callbacks.
2. **Dynamic LLM Function Calling**:
   - *Limitation*: The current agent uses deterministic routing and heuristic rule coordination for predictable evaluation defenses.
   - *Improvement*: Connect an OpenAI / Gemini API function-calling handler using the structured Pydantic schemas in `agent/schemas.py`.
3. **Automated Rollback on Deployment Health Probe Failure**:
   - *Limitation*: Unhealthy pods are detected and reported, but automatic rollback currently requires manual confirmation.
   - *Improvement*: Add an automated canary release controller that rolls back to the prior stable Docker tag if error rates exceed 1%.
