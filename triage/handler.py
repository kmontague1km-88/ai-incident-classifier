"""AWS Lambda entry point.

Trigger: SNS topic that receives CloudWatch alarms, GuardDuty findings (via
EventBridge), and AWS Budgets notifications. For each record the handler
extracts readable alert text, classifies it, publishes it to the owning
team's SNS topic (or the triage topic), and writes a row to DynamoDB so
every decision is auditable and can be used for retraining.
"""
import json
import time
import uuid

import boto3

from . import config as C
from .model import load
from .router import decide

_PIPE = None      # loaded once per container (warm starts reuse it)
_ROUTING = None


def _init():
    global _PIPE, _ROUTING
    if _PIPE is None:
        _PIPE = load()
        _ROUTING = C.load_routing()


def alert_text(message):
    """Normalize the different alert shapes into one line of text."""
    try:
        m = json.loads(message)
    except (TypeError, ValueError):
        return str(message)
    if "AlarmName" in m:  # CloudWatch alarm
        return " ".join(str(m.get(k, "")) for k in ("AlarmName", "AlarmDescription", "NewStateReason")).strip()
    if "detail" in m and isinstance(m["detail"], dict):  # EventBridge (GuardDuty etc.)
        d = m["detail"]
        return " ".join(str(x) for x in (m.get("detail-type", ""), d.get("type", ""), d.get("title", ""),
                                         d.get("description", "")) if x).strip()
    return m.get("message") or m.get("text") or json.dumps(m)


def handle(event, context=None, sns=None, table=None, bedrock=None):
    _init()
    sns = sns or boto3.client("sns")
    table = table or boto3.resource("dynamodb").Table(C.TRIAGE_TABLE)
    if bedrock is None and C.USE_BEDROCK:
        bedrock = boto3.client("bedrock-runtime")
    results = []
    for rec in event.get("Records", []):
        start = time.perf_counter()
        text = alert_text(rec.get("Sns", {}).get("Message", ""))
        d = decide(_PIPE, text, _ROUTING, bedrock_client=bedrock)
        alert_id = rec.get("Sns", {}).get("MessageId") or str(uuid.uuid4())
        body = (f"[{d['category']}] -> {d['team']} ({d['action']}, confidence {d['confidence']})\n"
                f"Alert: {text}\nEvidence: {', '.join(d['evidence']) or 'n/a'}\nRunbook:\n"
                + "\n".join(f"  {i}. {s}" for i, s in enumerate(d["runbook"], 1)))
        sns.publish(TopicArn=d["topic_arn"], Subject=f"[{d['category']}] alert routed to {d['team']}"[:100],
                    Message=body,
                    MessageAttributes={"category": {"DataType": "String", "StringValue": d["category"]},
                                       "action": {"DataType": "String", "StringValue": d["action"]}})
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        table.put_item(Item={"alert_id": alert_id, "text": text, "category": d["category"],
                             "confidence": str(d["confidence"]), "action": d["action"], "team": d["team"],
                             "source": d["source"], "latency_ms": str(latency_ms),
                             "ts": int(time.time())})
        results.append({**d, "alert_id": alert_id, "latency_ms": latency_ms})
    return {"processed": len(results), "results": results}


def lambda_handler(event, context):
    return handle(event, context)
