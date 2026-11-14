import os
import re
import hashlib
import subprocess
from typing import Dict, Any, List

try:
    import yaml
except ImportError:
    yaml = None


def _parse_yaml_file(filepath: str) -> Dict[str, Any]:
    if yaml is not None:
        with open(filepath, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    # fallback simple parser for dvc yaml structures
    data = {"outs": []}
    current_out = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str.startswith("- md5:") or line_str.startswith("md5:"):
                val = line_str.split(":", 1)[1].strip()
                current_out["md5"] = val
            elif line_str.startswith("size:"):
                val = line_str.split(":", 1)[1].strip()
                try:
                    current_out["size"] = int(val)
                except ValueError:
                    current_out["size"] = val
            elif line_str.startswith("path:"):
                val = line_str.split(":", 1)[1].strip()
                current_out["path"] = val
                data["outs"].append(dict(current_out))
                current_out = {}
    return data


APPROVED_DVC_COMMANDS = {"status", "diff", "checkout", "pull"}


def _run_cmd(cmd: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def _compute_md5(file_path: str) -> str:
    if not os.path.exists(file_path):
        return ""
    with open(file_path, "rb") as f:
        content = f.read()
    # normalize CRLF to LF for cross-platform consistency
    content = content.replace(b"\r\n", b"\n")
    return hashlib.md5(content).hexdigest()


def _get_git_commit() -> str:
    res = _run_cmd(["git", "rev-parse", "--short", "HEAD"])
    return res.stdout.strip() if res.returncode == 0 else "unknown"


def _get_git_branch() -> str:
    res = _run_cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    return res.stdout.strip() if res.returncode == 0 else "unknown"


def list_tracked_datasets(root_dir: str = ".") -> Dict[str, Any]:
    tracked = []
    # scan for .dvc files in root and data dirs
    for dirpath, _, filenames in os.walk(root_dir):
        if ".git" in dirpath or ".dvc" in dirpath:
            continue
        for fn in filenames:
            if fn.endswith(".dvc"):
                dvc_path = os.path.join(dirpath, fn)
                try:
                    content = _parse_yaml_file(dvc_path)
                    outs = content.get("outs", [])
                    for out in outs:
                        tracked.append({
                            "name": out.get("path", fn[:-4]),
                            "dvc_file": dvc_path,
                            "md5": out.get("md5"),
                            "size": out.get("size")
                        })
                except Exception as e:
                    return {"success": False, "error": f"Failed reading {dvc_path}: {e}"}

    # also check dvc.yaml
    dvc_yaml_path = os.path.join(root_dir, "dvc.yaml")
    if os.path.exists(dvc_yaml_path):
        try:
            dvc_yaml = _parse_yaml_file(dvc_yaml_path)
            stages = dvc_yaml.get("stages", {})
            for stage_name, stage_cfg in stages.items():
                if isinstance(stage_cfg, dict):
                    for dep in stage_cfg.get("deps", []):
                        if dep.startswith("data/"):
                            tracked.append({
                                "name": os.path.basename(dep),
                                "dvc_file": "dvc.yaml",
                                "stage": stage_name,
                                "path": dep
                            })
        except Exception:
            pass

    return {"success": True, "datasets": tracked}


def get_dataset_version(dataset_path: str = "data/dataset.csv") -> Dict[str, Any]:
    dvc_file = f"{dataset_path}.dvc"
    md5_hash = None

    if os.path.exists(dvc_file):
        try:
            data = _parse_yaml_file(dvc_file)
            outs = data.get("outs", [])
            if outs:
                md5_hash = outs[0].get("md5")
        except Exception as e:
            return {"success": False, "error": f"Error parsing {dvc_file}: {e}"}
    else:
        # compute current file hash if no .dvc file
        md5_hash = _compute_md5(dataset_path)

    commit = _get_git_commit()
    branch = _get_git_branch()

    return {
        "success": True,
        "dataset_name": os.path.basename(dataset_path),
        "dataset_path": dataset_path,
        "dvc_hash": md5_hash,
        "git_commit": commit,
        "git_branch": branch,
        "remote": "s3://mlops-dvc-storage/data"
    }


def compare_dataset_versions(version_a: str, version_b: str, dataset_path: str = "data/dataset.csv") -> Dict[str, Any]:
    if not version_a or not version_b:
        return {"success": False, "error": "Both version_a and version_b must be provided."}

    identical = (version_a.strip() == version_b.strip())
    summary = "Versions are identical." if identical else f"Version mismatch: {version_a} vs {version_b}."

    return {
        "success": True,
        "dataset_path": dataset_path,
        "version_a": version_a,
        "version_b": version_b,
        "identical": identical,
        "details": summary
    }


def detect_data_changes(dataset_path: str = "data/dataset.csv") -> Dict[str, Any]:
    if not os.path.exists(dataset_path):
        return {"success": False, "error": f"File {dataset_path} does not exist."}

    curr_md5 = _compute_md5(dataset_path)
    dvc_file = f"{dataset_path}.dvc"
    tracked_md5 = None

    if os.path.exists(dvc_file):
        try:
            data = _parse_yaml_file(dvc_file)
            outs = data.get("outs", [])
            if outs:
                tracked_md5 = outs[0].get("md5")
        except Exception as e:
            return {"success": False, "error": f"Could not read {dvc_file}: {e}"}

    changed = (curr_md5 != tracked_md5) if tracked_md5 else False
    status = "Data modified since last DVC commit" if changed else "Data matches tracked DVC version"

    return {
        "success": True,
        "dataset_path": dataset_path,
        "has_changed": changed,
        "current_md5": curr_md5,
        "tracked_md5": tracked_md5,
        "status_summary": status
    }


def explain_reproduction(dataset_name: str = "dataset.csv", target_version: str = "HEAD") -> Dict[str, Any]:
    instructions = [
        f"1. Fetch Git history and checkout target commit: git checkout {target_version}",
        f"2. Pull the associated DVC pointers and dataset: dvc pull data/{dataset_name}.dvc",
        "3. Run 'dvc status' to verify working tree data matches the recorded hash."
    ]
    return {
        "success": True,
        "dataset_name": dataset_name,
        "target_version": target_version,
        "instructions": instructions,
        "git_command": f"git checkout {target_version}",
        "dvc_command": f"dvc pull data/{dataset_name}.dvc"
    }


def checkout_dataset_version(target_commit: str, dataset_path: str = "data/dataset.csv") -> Dict[str, Any]:
    # safety guard on git ref to prevent command injection
    if not re.match(r"^[a-zA-Z0-9_\.\-]+$", target_commit):
        return {"success": False, "error": "Invalid target commit/ref format."}

    git_res = _run_cmd(["git", "checkout", target_commit, "--", f"{dataset_path}.dvc"])
    if git_res.returncode != 0:
        return {"success": False, "error": git_res.stderr.strip() or "Git checkout failed."}

    return {
        "success": True,
        "message": f"Checked out {dataset_path}.dvc at ref {target_commit}",
        "target_commit": target_commit
    }
