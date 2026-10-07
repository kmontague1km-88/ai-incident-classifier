from triage import config as C
from triage.router import decide


class FakeBedrock:
    """Stands in for the bedrock-runtime client."""
    def __init__(self, answer):
        self.answer, self.calls = answer, 0

    def converse(self, modelId, messages, inferenceConfig):
        self.calls += 1
        assert "Classify this cloud monitoring alert" in messages[0]["content"][0]["text"]
        return {"output": {"message": {"content": [{"text": self.answer}]}}}


def test_high_confidence_auto_routes_with_runbook(pipe):
    d = decide(pipe, "ALARM: DatabaseConnections > 400 on rds-prod-main", C.load_routing())
    assert d["action"] == "AUTO_ROUTE" and d["team"] == "Database Administration"
    assert len(d["runbook"]) == 3 and d["evidence"]


def test_low_confidence_goes_to_human_triage(pipe):
    d = decide(pipe, "something looks weird, please check", C.load_routing())
    assert d["action"] == "HUMAN_TRIAGE"
    assert d["topic_arn"] == C.TRIAGE_QUEUE_TOPIC
    assert d["category"] in C.CATEGORIES  # still offers its best guess to the human


def test_threshold_boundary(pipe):
    text = "ALARM: DatabaseConnections > 400 on rds-prod-main"
    conf = decide(pipe, text, C.load_routing())["confidence"]
    assert decide(pipe, text, C.load_routing(), threshold=conf)["action"] == "AUTO_ROUTE"
    assert decide(pipe, text, C.load_routing(), threshold=min(conf + 0.01, 1.0))["action"] == "HUMAN_TRIAGE"


def test_llm_agreement_promotes_to_auto_route(pipe):
    text = "lock waits piling up on orders-db"
    base = decide(pipe, text, C.load_routing(), threshold=0.99)
    fake = FakeBedrock(base["category"])
    d = decide(pipe, text, C.load_routing(), threshold=0.99, bedrock_client=fake)
    assert fake.calls == 1 and d["action"] == "AUTO_ROUTE" and d["source"] == "model+llm"


def test_llm_disagreement_stays_with_human(pipe):
    text = "lock waits piling up on orders-db"
    base = decide(pipe, text, C.load_routing(), threshold=0.99)
    other = next(c for c in C.CATEGORIES if c != base["category"])
    d = decide(pipe, text, C.load_routing(), threshold=0.99, bedrock_client=FakeBedrock(other))
    assert d["action"] == "HUMAN_TRIAGE" and d["llm_suggestion"] == other


def test_llm_invalid_answer_is_ignored(pipe):
    d = decide(pipe, "lock waits piling up on orders-db", C.load_routing(), threshold=0.99,
               bedrock_client=FakeBedrock("I think it is probably the database"))
    assert d["action"] == "HUMAN_TRIAGE" and d["llm_suggestion"] is None
