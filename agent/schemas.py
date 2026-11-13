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


