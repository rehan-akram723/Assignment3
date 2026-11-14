from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class ToolResult(BaseModel):
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class DvcDatasetInfo(BaseModel):
    name: str
    path: str
    dvc_file: Optional[str] = None
    md5: Optional[str] = None
    size: Optional[int] = None


class DvcVersionInfo(BaseModel):
    dataset_name: str
    dataset_path: str
    dvc_hash: Optional[str] = None
    git_commit: Optional[str] = None
    git_branch: Optional[str] = None
    remote: Optional[str] = None


class DvcDiffResult(BaseModel):
    identical: bool
    version_a: str
    version_b: str
    details: str


class DvcStatusResult(BaseModel):
    dataset_path: str
    has_changed: bool
    current_md5: Optional[str] = None
    tracked_md5: Optional[str] = None
    status_summary: str


class DvcReproductionGuide(BaseModel):
    dataset_name: str
    target_version: str
    instructions: List[str]
    git_command: str
    dvc_command: str


class MlflowRunSummary(BaseModel):
    run_id: str
    run_name: Optional[str] = None
    status: str
    start_time: Optional[int] = None
    metrics: Dict[str, float] = Field(default_factory=dict)
    params: Dict[str, str] = Field(default_factory=dict)
    tags: Dict[str, str] = Field(default_factory=dict)


class MlflowBestRunResult(BaseModel):
    best_run_id: str
    metric_name: str
    metric_value: float
    reasoning: str
    is_production_ready: bool
    warnings: List[str] = Field(default_factory=list)


class MlflowComparison(BaseModel):
    runs: List[Dict[str, Any]]
    metrics_compared: List[str]
    analysis: str


class ModelRegistryInfo(BaseModel):
    model_name: str
    latest_versions: List[Dict[str, Any]]


class AirflowDagSummary(BaseModel):
    dag_id: str
    description: Optional[str] = None
    is_paused: bool
    is_active: bool


class AirflowDagRunStatus(BaseModel):
    dag_id: str
    dag_run_id: str
    state: str
    execution_date: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None


class AirflowTriggerResult(BaseModel):
    dag_id: str
    dag_run_id: str
    message: str
    state: str


class AirflowTaskFailure(BaseModel):
    dag_id: str
    dag_run_id: str
    task_id: str
    state: str
    try_number: int
    start_date: Optional[str] = None
    end_date: Optional[str] = None


class AirflowFailureDiagnostics(BaseModel):
    dag_id: str
    dag_run_id: str
    task_id: str
    error_message: str
    logs_snippet: str
    recommended_fix: str


class K8sPodSummary(BaseModel):
    name: str
    namespace: str
    phase: str
    ready: bool
    restarts: int
    pod_ip: Optional[str] = None
    node_name: Optional[str] = None


class K8sDeploymentSummary(BaseModel):
    name: str
    namespace: str
    replicas: int
    ready_replicas: int
    updated_replicas: int
    available_replicas: int
    image: str


class K8sServiceSummary(BaseModel):
    name: str
    namespace: str
    type: str
    cluster_ip: str
    ports: List[Dict[str, Any]]


class K8sUnhealthyPod(BaseModel):
    name: str
    namespace: str
    phase: str
    reason: str
    restart_count: int
    message: str


class K8sClusterOperationResult(BaseModel):
    success: bool
    operation: str
    target: str
    requires_confirmation: bool = False
    message: str


class EvidenceRecord(BaseModel):
    tool_name: str
    category: str
    success: bool
    duration_ms: float
    data: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


class AgentResponse(BaseModel):
    query: str
    answer: str
    known_facts: List[str]
    uncertainties: List[str]
    recommended_actions: List[str]
    retraining_recommended: bool
    retraining_justification: Optional[str] = None
    evidence: List[EvidenceRecord]





