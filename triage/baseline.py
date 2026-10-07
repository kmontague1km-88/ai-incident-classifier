"""The EXISTING automation: a hand-maintained keyword router.

This is the workflow being enhanced. Rules are checked in order and the first
match wins; anything unmatched lands in a general queue (UNROUTED). It is fast
and transparent but brittle: new wording or vendor jargon falls through.
"""
import re

RULES = [
    ("SECURITY", r"guardduty|unauthorized|brute ?force|failed (console )?login|iam policy|cloudtrail|waf|root account|public|security group"),
    ("COST_BILLING", r"budget|cost|billing|spend|charges|savings plan|reserved instance"),
    ("DATABASE", r"database|rds|aurora|dynamodb|replica|deadlock|query|connections"),
    ("STORAGE", r"disk|ebs|filesystem|s3 bucket|efs|snapshot|inode|partition"),
    ("NETWORK", r"latency|packet loss|vpn|dns|route 53|load balancer|nat gateway|direct connect|network"),
    ("APPLICATION_DEPLOY", r"5xx|deploy|pipeline|exception|503|crashloop|error rate|canary"),
    ("COMPUTE_CAPACITY", r"cpu|memory|auto ?scaling|instance|oom|throttled|node"),
]
_COMPILED = [(label, re.compile(p, re.IGNORECASE)) for label, p in RULES]
UNROUTED = "UNROUTED"


def classify(text):
    for label, rx in _COMPILED:
        if rx.search(text):
            return label
    return UNROUTED
