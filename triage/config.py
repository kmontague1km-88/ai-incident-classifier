"""Central configuration. Every value can be overridden with an environment
variable so thresholds and routing change without a code change."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CATEGORIES = [
    "COMPUTE_CAPACITY",    # CPU, memory, autoscaling, instance health
    "NETWORK",             # latency, packet loss, DNS, load balancer, VPN
    "DATABASE",            # RDS/DynamoDB connections, replication, deadlocks
    "STORAGE",             # disk/EBS/S3 capacity and I/O
    "SECURITY",            # GuardDuty, IAM, failed logins, WAF
    "APPLICATION_DEPLOY",  # 5xx errors, failed deployments, crashes
    "COST_BILLING",        # budget and spend anomalies
]

# Below this confidence the alert goes to the human triage queue instead of
# being auto-routed. Chosen by leave-templates-out cross-validation (see evaluate.py).
CONFIDENCE_THRESHOLD = float(os.environ.get("TRIAGE_CONFIDENCE_THRESHOLD", "0.65"))

MODEL_PATH = Path(os.environ.get("TRIAGE_MODEL_PATH", ROOT / "model" / "triage_model.joblib"))
ROUTING_FILE = Path(os.environ.get("TRIAGE_ROUTING_FILE", ROOT / "config" / "routing.json"))

TRIAGE_TABLE = os.environ.get("TRIAGE_TABLE", "alert-triage-log")
TRIAGE_QUEUE_TOPIC = os.environ.get(
    "TRIAGE_QUEUE_TOPIC", "arn:aws:sns:us-east-1:123456789012:triage-human-review")

# Optional second opinion from a foundation model on Amazon Bedrock for
# low-confidence alerts. Off by default (cost and latency).
USE_BEDROCK = os.environ.get("TRIAGE_USE_BEDROCK", "false").lower() == "true"
BEDROCK_MODEL_ID = os.environ.get("TRIAGE_BEDROCK_MODEL_ID", "anthropic.claude-3-haiku-20240307-v1:0")


def load_routing():
    """Category -> {team, topic_arn, runbook}. Topic ARNs can be overridden
    per category with TRIAGE_TOPIC_<CATEGORY>."""
    with open(ROUTING_FILE, encoding="utf-8") as fh:
        routing = json.load(fh)
    for cat, entry in routing.items():
        entry["topic_arn"] = os.environ.get(f"TRIAGE_TOPIC_{cat}", entry["topic_arn"])
    return routing
