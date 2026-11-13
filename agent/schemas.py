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
