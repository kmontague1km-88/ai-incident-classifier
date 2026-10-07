import numpy as np

from triage import config as C
from triage.model import explain, predict


def test_predicts_every_category_on_clear_examples(pipe):
    cases = {
        "CPUUtilization above 95 percent on api-prod-3": "COMPUTE_CAPACITY",
        "VPN tunnel to corporate network is DOWN": "NETWORK",
        "Deadlock detected in orders-db": "DATABASE",
        "Filesystem / is 97% full on web-07": "STORAGE",
        "GuardDuty finding SSHBruteForce from 203.0.113.45": "SECURITY",
        "Deployment v2.3 of checkout failed health checks": "APPLICATION_DEPLOY",
        "Cost anomaly detected: EC2 spend up $4000 today": "COST_BILLING",
    }
    for text, label in cases.items():
        assert predict(pipe, text).category == label, text


def test_generalizes_to_unseen_wording(pipe, novel):
    texts, labels = novel
    preds = pipe.predict(texts)
    acc = float(np.mean(np.array(preds) == np.array(labels)))
    assert acc >= 0.75, f"novel-phrasing accuracy dropped to {acc:.3f}"


def test_confidence_is_a_probability(pipe):
    p = predict(pipe, "something odd happened")
    assert 0.0 < p.confidence <= 1.0
    assert p.category in C.CATEGORIES and p.runner_up in C.CATEGORIES


def test_explanation_terms_come_from_the_alert(pipe):
    text = "Replica lag on customers-aurora is 300 seconds"
    terms = explain(pipe, text, "DATABASE")
    assert terms and all(any(w in text.lower() for w in t.split()) for t in terms)


def test_deterministic_training(pipe):
    from triage import dataset, model
    rows, _ = dataset.build()
    again = model.train([r[0] for r in rows], [r[1] for r in rows])
    t = "Too many connections error from application to orders-db"
    assert predict(pipe, t).confidence == predict(again, t).confidence
