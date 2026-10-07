import json

from triage import handler


def _event(*messages):
    return {"Records": [{"Sns": {"MessageId": f"msg-{i}", "Message": m}} for i, m in enumerate(messages)]}


def _drain(aws, queue):
    out = aws["sqs"].receive_message(QueueUrl=aws["queues"][queue], MaxNumberOfMessages=10)
    return [json.loads(m["Body"]) for m in out.get("Messages", [])]


def test_alert_text_normalizes_cloudwatch_and_eventbridge():
    cw = json.dumps({"AlarmName": "orders-db-connections", "AlarmDescription": "DatabaseConnections high",
                     "NewStateReason": "Threshold Crossed: 1 datapoint [412.0] > 400"})
    assert "DatabaseConnections high" in handler.alert_text(cw)
    gd = json.dumps({"detail-type": "GuardDuty Finding",
                     "detail": {"type": "UnauthorizedAccess:EC2/SSHBruteForce", "title": "SSH brute force"}})
    assert "SSHBruteForce" in handler.alert_text(gd)
    assert handler.alert_text("plain text alert") == "plain text alert"


def test_end_to_end_routing_delivery_and_audit(aws, pipe, monkeypatch):
    monkeypatch.setattr(handler, "_PIPE", pipe)
    monkeypatch.setattr(handler, "_ROUTING", handler.C.load_routing())
    cw = json.dumps({"AlarmName": "orders-db-connections", "AlarmDescription": "DatabaseConnections > 400 on orders-db",
                     "NewStateReason": "Threshold Crossed"})
    out = handler.handle(_event(cw, "something looks weird, please check"), sns=aws["sns"], table=aws["table"])

    assert out["processed"] == 2
    assert [r["action"] for r in out["results"]] == ["AUTO_ROUTE", "HUMAN_TRIAGE"]
    dba = _drain(aws, "team-dba")
    assert len(dba) == 1 and dba[0]["MessageAttributes"]["category"]["Value"] == "DATABASE"
    assert "Runbook" in dba[0]["Message"]
    assert len(_drain(aws, "triage-human-review")) == 1
    assert _drain(aws, "team-secops") == []  # nothing leaked to other teams

    item = aws["table"].get_item(Key={"alert_id": "msg-0"})["Item"]
    assert item["category"] == "DATABASE" and item["action"] == "AUTO_ROUTE"


def test_bedrock_not_called_when_disabled(aws, pipe, monkeypatch):
    monkeypatch.setattr(handler, "_PIPE", pipe)
    monkeypatch.setattr(handler, "_ROUTING", handler.C.load_routing())
    monkeypatch.setattr(handler.C, "USE_BEDROCK", False)
    out = handler.handle(_event("something looks weird, please check"), sns=aws["sns"], table=aws["table"])
    assert "llm_suggestion" not in out["results"][0]
