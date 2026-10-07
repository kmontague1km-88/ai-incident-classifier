import os
import sys
from pathlib import Path

import boto3
import pytest
from moto import mock_aws

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "testing")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "testing")

from triage import config as C  # noqa: E402
from triage import dataset, model  # noqa: E402


@pytest.fixture(scope="session")
def pipe():
    """Train once per test session on the training templates only."""
    rows, _ = dataset.build()
    return model.train([r[0] for r in rows], [r[1] for r in rows])


@pytest.fixture(scope="session")
def novel():
    _, rows = dataset.build()
    return [r[0] for r in rows], [r[1] for r in rows]


@pytest.fixture
def aws():
    """Mocked AWS: one SNS topic per team plus the triage topic, each with an
    SQS subscription so tests can read what was actually delivered."""
    with mock_aws():
        sns, sqs = boto3.client("sns"), boto3.client("sqs")
        ddb = boto3.resource("dynamodb")
        table = ddb.create_table(TableName=C.TRIAGE_TABLE, BillingMode="PAY_PER_REQUEST",
                                 KeySchema=[{"AttributeName": "alert_id", "KeyType": "HASH"}],
                                 AttributeDefinitions=[{"AttributeName": "alert_id", "AttributeType": "S"}])
        routing = C.load_routing()
        queues = {}
        names = {e["topic_arn"].split(":")[-1] for e in routing.values()} | {C.TRIAGE_QUEUE_TOPIC.split(":")[-1]}
        for name in names:
            arn = sns.create_topic(Name=name)["TopicArn"]
            q = sqs.create_queue(QueueName=name)["QueueUrl"]
            qarn = sqs.get_queue_attributes(QueueUrl=q, AttributeNames=["QueueArn"])["Attributes"]["QueueArn"]
            sns.subscribe(TopicArn=arn, Protocol="sqs", Endpoint=qarn)
            queues[name] = q
        yield {"sns": sns, "sqs": sqs, "table": table, "queues": queues}
