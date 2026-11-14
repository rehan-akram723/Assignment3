import pytest
from agent.tools import dvc_tools


def test_list_tracked_datasets():
    res = dvc_tools.list_tracked_datasets()
    assert res["success"] is True
    assert "datasets" in res
    dataset_names = [d["name"] for d in res["datasets"]]
    assert "dataset.csv" in dataset_names


def test_get_dataset_version():
    res = dvc_tools.get_dataset_version("data/dataset.csv")
    assert res["success"] is True
    assert res["dvc_hash"] == "52aebc1284c804dab986501dcf8c8efe"
    assert res["git_commit"] != "unknown"


def test_detect_data_changes_unmodified():
    res = dvc_tools.detect_data_changes("data/dataset.csv")
    assert res["success"] is True
    assert res["has_changed"] is False
    assert res["current_md5"] == res["tracked_md5"]


def test_checkout_dataset_security_guard():
    # invalid characters should be blocked
    malicious_ref = "HEAD; rm -rf /"
    res = dvc_tools.checkout_dataset_version(malicious_ref)
    assert res["success"] is False
    assert "Invalid target commit/ref" in res["error"]
