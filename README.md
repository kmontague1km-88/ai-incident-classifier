# Alert Triage AI

AI enhancement for an existing cloud alert-routing workflow (CLCS 660, Unit 6,
Scenario 2: Automated Incident Classification).

Repository: https://github.com/kmontague1km-88/ai-incident-classifier

**Before:** a keyword rule list routes CloudWatch, GuardDuty, and AWS Budgets
alerts to teams. New wording falls through to a general queue or matches the
wrong rule.

**After:** a Lambda function classifies each alert with an NLP model (TF-IDF +
logistic regression), routes confident predictions straight to the owning
team's SNS topic with a runbook, and sends low-confidence alerts to a human
triage queue with the model's best guess and supporting evidence. An optional
Amazon Bedrock foundation model gives a second opinion on low-confidence alerts.
Every decision is written to DynamoDB for audit and retraining.

## Layout

| Path | Purpose |
|---|---|
| `triage/baseline.py` | The existing keyword router (the workflow being enhanced) |
| `triage/model.py` | Feature pipeline, training, prediction, per-word explanation |
| `triage/router.py` | Confidence threshold, routing decision, Bedrock second opinion |
| `triage/handler.py` | AWS Lambda entry point (SNS in -> SNS out + DynamoDB log) |
| `triage/dataset.py` | Synthetic labeled alerts (training templates + novel-phrasing set) |
| `triage/config.py`, `config/routing.json` | Thresholds, team topics, runbooks |
| `evaluate.py` | Trains the model, picks the threshold, compares against the baseline |
| `tests/` | 17 pytest tests; AWS mocked with moto |
| `deploy/template.yaml`, `Dockerfile` | AWS SAM template and Lambda container image |
| `results/` | Metrics, test output, Bandit scan |

## Run locally (no AWS account needed)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
python triage/dataset.py          # writes data/alerts.csv and data/alerts_novel_phrasing.csv
python evaluate.py                # trains model/triage_model.joblib and writes results/
python -m pytest -v --cov=triage  # 17 tests
bandit -r triage                  # static security scan
```

## Deploy to AWS

```bash
python evaluate.py                # make sure model/triage_model.joblib exists
sam build -t deploy/template.yaml
sam deploy --guided --parameter-overrides TriageEmail=oncall@example.com ConfidenceThreshold=0.65 UseBedrock=false
```

Then point CloudWatch alarm actions, AWS Budgets notifications, and an
EventBridge rule for GuardDuty findings at the `IncomingAlertsTopicArn` output,
and subscribe each team (email, Slack via AWS Chatbot, or PagerDuty) to its
`team-*` topic. Test with `sam local invoke TriageFunction -e deploy/sample-events/cloudwatch-alarm.json`.

To enable the Bedrock second opinion, request model access for Amazon Nova Micro
in the Bedrock console and redeploy with `UseBedrock=true`.

## Retraining

Analysts who correct a triaged alert record the right category in the
`alert-triage-log` table. Export those rows, append them to the training data,
rerun `evaluate.py`, and redeploy only if the novel-phrasing scores do not drop.
