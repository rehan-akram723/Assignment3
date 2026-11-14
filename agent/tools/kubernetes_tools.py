import json
import subprocess
from typing import Dict, Any, List, Optional

APPROVED_NAMESPACES = {"default", "mlops"}


def _run_kubectl(args: List[str]) -> subprocess.CompletedProcess:
    cmd = ["kubectl"] + args
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def list_pods(namespace: str = "mlops", label_selector: Optional[str] = "app=churn-inference-api") -> Dict[str, Any]:
    cmd = ["get", "pods", "-n", namespace, "-o", "json"]
    if label_selector:
        cmd.extend(["-l", label_selector])

    res = _run_kubectl(cmd)
    if res.returncode != 0:
        return {
            "success": False,
            "error": f"Failed listing pods: {res.stderr.strip() or 'Kubernetes cluster unreachable'}",
            "cluster_available": False
        }

    try:
        data = json.loads(res.stdout)
        pods = []
        for item in data.get("items", []):
            metadata = item.get("metadata", {})
            status = item.get("status", {})
            c_statuses = status.get("containerStatuses", [])

            restarts = sum(cs.get("restartCount", 0) for cs in c_statuses)
            ready = all(cs.get("ready", False) for cs in c_statuses) if c_statuses else False

            pods.append({
                "name": metadata.get("name"),
                "namespace": metadata.get("namespace"),
                "phase": status.get("phase"),
                "ready": ready,
                "restarts": restarts,
                "pod_ip": status.get("podIP"),
                "node_name": item.get("spec", {}).get("nodeName")
            })
        return {"success": True, "namespace": namespace, "pods": pods}
    except Exception as e:
        return {"success": False, "error": f"Failed parsing kubectl output: {str(e)}"}


def list_deployments(namespace: str = "mlops") -> Dict[str, Any]:
    res = _run_kubectl(["get", "deployments", "-n", namespace, "-o", "json"])
    if res.returncode != 0:
        return {
            "success": False,
            "error": f"Failed listing deployments: {res.stderr.strip() or 'Kubernetes cluster unreachable'}",
            "cluster_available": False
        }

    try:
        data = json.loads(res.stdout)
        deployments = []
        for item in data.get("items", []):
            metadata = item.get("metadata", {})
            spec = item.get("spec", {})
            status = item.get("status", {})

            containers = spec.get("template", {}).get("spec", {}).get("containers", [])
            image = containers[0].get("image") if containers else "unknown"

            deployments.append({
                "name": metadata.get("name"),
                "namespace": metadata.get("namespace"),
                "replicas": spec.get("replicas", 0),
                "ready_replicas": status.get("readyReplicas", 0),
                "updated_replicas": status.get("updatedReplicas", 0),
                "available_replicas": status.get("availableReplicas", 0),
                "image": image
            })
        return {"success": True, "namespace": namespace, "deployments": deployments}
    except Exception as e:
        return {"success": False, "error": f"Failed parsing deployments: {str(e)}"}


def get_pod_status_and_logs(pod_name: str, namespace: str = "mlops", tail_lines: int = 50) -> Dict[str, Any]:
    pod_res = _run_kubectl(["get", "pod", pod_name, "-n", namespace, "-o", "json"])
    if pod_res.returncode != 0:
        return {"success": False, "error": f"Pod '{pod_name}' not found: {pod_res.stderr.strip()}"}

    try:
        pod_data = json.loads(pod_res.stdout)
        status = pod_data.get("status", {})
        phase = status.get("phase")
        container_statuses = status.get("containerStatuses", [])
    except Exception as e:
        return {"success": False, "error": f"Error parsing pod JSON: {e}"}

    log_res = _run_kubectl(["logs", pod_name, "-n", namespace, f"--tail={tail_lines}"])
    logs = log_res.stdout if log_res.returncode == 0 else log_res.stderr

    return {
        "success": True,
        "pod_name": pod_name,
        "namespace": namespace,
        "phase": phase,
        "container_statuses": container_statuses,
        "logs": logs
    }


def inspect_service(service_name: str = "churn-inference-service", namespace: str = "mlops") -> Dict[str, Any]:
    res = _run_kubectl(["get", "service", service_name, "-n", namespace, "-o", "json"])
    if res.returncode != 0:
        return {"success": False, "error": f"Service '{service_name}' not found: {res.stderr.strip()}"}

    try:
        svc_data = json.loads(res.stdout)
        spec = svc_data.get("spec", {})
        return {
            "success": True,
            "service_name": service_name,
            "namespace": namespace,
            "type": spec.get("type"),
            "cluster_ip": spec.get("clusterIP"),
            "ports": spec.get("ports", []),
            "selector": spec.get("selector", {})
        }
    except Exception as e:
        return {"success": False, "error": f"Error parsing service JSON: {e}"}


def detect_unhealthy_pods(namespace: str = "mlops") -> Dict[str, Any]:
    pods_res = list_pods(namespace=namespace)
    if not pods_res.get("success"):
        return pods_res

    pods = pods_res.get("pods", [])
    unhealthy = []

    for p in pods:
        reasons = []
        if p.get("phase") != "Running":
            reasons.append(f"Phase is {p.get('phase')}")
        if not p.get("ready"):
            reasons.append("Readiness probe failing")
        if p.get("restarts", 0) > 0:
            reasons.append(f"Restart count is {p.get('restarts')}")

        if reasons:
            unhealthy.append({
                "name": p.get("name"),
                "namespace": namespace,
                "phase": p.get("phase"),
                "restarts": p.get("restarts"),
                "diagnostic_reasons": reasons
            })

    return {
        "success": True,
        "namespace": namespace,
        "total_pods": len(pods),
        "unhealthy_count": len(unhealthy),
        "unhealthy_pods": unhealthy
    }


def get_deployed_application_version(deployment_name: str = "churn-inference-deployment", namespace: str = "mlops") -> Dict[str, Any]:
    res = _run_kubectl(["get", "deployment", deployment_name, "-n", namespace, "-o", "json"])
    if res.returncode != 0:
        return {
            "success": False,
            "error": f"Deployment '{deployment_name}' not found: {res.stderr.strip()}",
            "cluster_available": False
        }

    try:
        dep_data = json.loads(res.stdout)
        containers = dep_data.get("spec", {}).get("template", {}).get("spec", {}).get("containers", [])
        image = containers[0].get("image", "") if containers else ""
        version_tag = image.split(":")[-1] if ":" in image else "unknown"

        return {
            "success": True,
            "deployment_name": deployment_name,
            "namespace": namespace,
            "image": image,
            "deployed_version": version_tag,
            "replicas": dep_data.get("spec", {}).get("replicas"),
            "ready_replicas": dep_data.get("status", {}).get("readyReplicas", 0)
        }
    except Exception as e:
        return {"success": False, "error": f"Failed reading deployment version: {e}"}


def restart_deployment(deployment_name: str = "churn-inference-deployment", namespace: str = "mlops", confirmed: bool = False) -> Dict[str, Any]:
    # safety: namespace validation
    if namespace not in APPROVED_NAMESPACES:
        return {
            "success": False,
            "error": f"Unauthorized namespace '{namespace}'. Allowed namespaces: {list(APPROVED_NAMESPACES)}."
        }

    # safety: explicit confirmation requirement
    if not confirmed:
        return {
            "success": False,
            "requires_confirmation": True,
            "operation": "restart_deployment",
            "target": f"{namespace}/{deployment_name}",
            "message": "Explicit user confirmation required before restarting deployment in cluster."
        }

    res = _run_kubectl(["rollout", "restart", f"deployment/{deployment_name}", "-n", namespace])
    if res.returncode != 0:
        return {"success": False, "error": f"Rollout restart failed: {res.stderr.strip()}"}

    return {
        "success": True,
        "operation": "restart_deployment",
        "target": f"{namespace}/{deployment_name}",
        "message": f"Deployment '{deployment_name}' restarted successfully."
    }


def scale_deployment(deployment_name: str = "churn-inference-deployment", replicas: int = 2, namespace: str = "mlops", confirmed: bool = False) -> Dict[str, Any]:
    # safety: namespace validation
    if namespace not in APPROVED_NAMESPACES:
        return {
            "success": False,
            "error": f"Unauthorized namespace '{namespace}'. Allowed namespaces: {list(APPROVED_NAMESPACES)}."
        }

    # safety: explicit confirmation requirement
    if not confirmed:
        return {
            "success": False,
            "requires_confirmation": True,
            "operation": "scale_deployment",
            "target": f"{namespace}/{deployment_name}",
            "requested_replicas": replicas,
            "message": f"Explicit user confirmation required to scale deployment to {replicas} replicas."
        }

    res = _run_kubectl(["scale", f"deployment/{deployment_name}", f"--replicas={replicas}", "-n", namespace])
    if res.returncode != 0:
        return {"success": False, "error": f"Scale failed: {res.stderr.strip()}"}

    return {
        "success": True,
        "operation": "scale_deployment",
        "target": f"{namespace}/{deployment_name}",
        "replicas": replicas,
        "message": f"Deployment '{deployment_name}' scaled to {replicas} replicas."
    }
