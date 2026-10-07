"""Decision layer: turns a prediction into a routing decision.

* confidence >= threshold -> auto-route to the owning team with a runbook
* confidence <  threshold -> human triage queue (optionally after asking a
  foundation model on Amazon Bedrock for a second opinion)
"""
from . import config as C
from .model import predict


def bedrock_second_opinion(client, text):
    """Zero-shot classification with a foundation model. Returns a category
    or None. The prompt restricts the answer to the known labels."""
    prompt = ("Classify this cloud monitoring alert into exactly one category: "
              + ", ".join(C.CATEGORIES)
              + ". Reply with the category name only.\n\nAlert: " + text)
    resp = client.converse(modelId=C.BEDROCK_MODEL_ID,
                           messages=[{"role": "user", "content": [{"text": prompt}]}],
                           inferenceConfig={"maxTokens": 20, "temperature": 0})
    answer = resp["output"]["message"]["content"][0]["text"].strip().upper()
    return answer if answer in C.CATEGORIES else None


def decide(pipe, text, routing, threshold=None, bedrock_client=None):
    threshold = C.CONFIDENCE_THRESHOLD if threshold is None else threshold
    p = predict(pipe, text)
    decision = {
        "category": p.category, "confidence": round(p.confidence, 3), "runner_up": p.runner_up,
        "evidence": p.top_terms, "source": "model",
    }
    if p.confidence >= threshold:
        r = routing[p.category]
        decision.update(action="AUTO_ROUTE", team=r["team"], topic_arn=r["topic_arn"], runbook=r["runbook"])
        return decision

    # Low confidence: never guess silently.
    if bedrock_client is not None:
        llm = bedrock_second_opinion(bedrock_client, text)
        decision["llm_suggestion"] = llm
        if llm == p.category:  # two independent methods agree -> route
            r = routing[llm]
            decision.update(action="AUTO_ROUTE", team=r["team"], topic_arn=r["topic_arn"],
                            runbook=r["runbook"], source="model+llm")
            return decision
    decision.update(action="HUMAN_TRIAGE", team="Triage On-Call", topic_arn=C.TRIAGE_QUEUE_TOPIC,
                    runbook=["Review the suggested category and evidence", "Assign to the correct team",
                             "Record the correct label so the model can be retrained"])
    return decision
