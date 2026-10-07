"""Train the model and measure the enhancement against the keyword baseline.

Usage:  python evaluate.py
Writes: model/triage_model.joblib, results/metrics.json, results/report.md
"""
import json
import statistics
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold, train_test_split

from triage import baseline, dataset, model
from triage import config as C

OUT = Path("results")
SEED = 42


def baseline_metrics(texts, labels):
    preds = [baseline.classify(t) for t in texts]
    unrouted = sum(p == baseline.UNROUTED for p in preds)
    wrong_routed = sum(p != y and p != baseline.UNROUTED for p, y in zip(preds, labels))
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, labels=C.CATEGORIES, average="macro", zero_division=0),
        "unrouted_rate": unrouted / len(labels),
        "misroute_rate": wrong_routed / len(labels),
    }


def model_metrics(pipe, texts, labels, threshold):
    proba = pipe.predict_proba(texts)
    classes = pipe.classes_
    preds = classes[proba.argmax(1)]
    conf = proba.max(1)
    auto = conf >= threshold
    y = np.array(labels)
    return {
        "accuracy": accuracy_score(y, preds),
        "macro_f1": f1_score(y, preds, labels=C.CATEGORIES, average="macro", zero_division=0),
        "auto_route_rate": float(auto.mean()),
        "auto_route_accuracy": float((preds[auto] == y[auto]).mean()) if auto.any() else 0.0,
        "misroute_rate": float(((preds != y) & auto).mean()),
        "human_triage_rate": float((~auto).mean()),
        "_preds": preds,
    }


def pick_threshold(texts, labels, groups, max_misroute=0.04):
    """Choose the confidence threshold with leave-templates-out
    cross-validation: each fold hides whole alert templates, so the
    out-of-fold predictions behave like alerts worded in a new way. The
    chosen threshold is the lowest (most automation) whose out-of-fold
    misroute rate stays at or below max_misroute (4%, the keyword router's
    misroute rate today). The novel-phrasing test
    set is never touched here."""
    texts, labels = np.array(texts, dtype=object), np.array(labels)
    proba = np.zeros((len(texts), len(C.CATEGORIES)))
    for tr, va in GroupKFold(n_splits=5).split(texts, labels, groups):
        pipe = model.train(list(texts[tr]), list(labels[tr]))
        cols = [list(pipe.classes_).index(c) for c in C.CATEGORIES]
        proba[va] = pipe.predict_proba(list(texts[va]))[:, cols]
    preds = np.array(C.CATEGORIES)[proba.argmax(1)]
    conf = proba.max(1)
    sweep = []
    for t in np.round(np.arange(0.30, 0.91, 0.05), 2):
        auto = conf >= t
        sweep.append({"threshold": float(t), "auto_route_rate": round(float(auto.mean()), 3),
                      "misroute_rate": round(float(((preds != labels) & auto).mean()), 3)})
    ok = [s for s in sweep if s["misroute_rate"] <= max_misroute]
    return (ok[0]["threshold"] if ok else 0.9), sweep


def main():
    main_rows, novel_rows = dataset.build()
    X = [r[0] for r in main_rows]
    y = [r[1] for r in main_rows]
    Xn = [r[0] for r in novel_rows]
    yn = [r[1] for r in novel_rows]
    G = [r[2] for r in main_rows]
    X_tr, X_te, y_tr, y_te, g_tr, _ = train_test_split(X, y, G, test_size=0.30, stratify=y, random_state=SEED)

    # 1) choose the confidence threshold on training data only (leave-templates-out CV)
    threshold, sweep = pick_threshold(X_tr, y_tr, g_tr)

    # 2) retrain on the full training split and save the deployable model
    t0 = time.perf_counter()
    pipe = model.train(X_tr, y_tr)
    train_s = time.perf_counter() - t0
    model.save(pipe)

    # 3) evaluate
    results = {"sizes": {"train": len(X_tr), "test": len(X_te), "novel": len(Xn)},
               "threshold": threshold, "threshold_sweep_cv": sweep, "train_seconds": round(train_s, 2)}
    for name, (xs, ys) in {"standard_test": (X_te, y_te), "novel_phrasing": (Xn, yn)}.items():
        b = baseline_metrics(xs, ys)
        m = model_metrics(pipe, xs, ys, threshold)
        preds = m.pop("_preds")
        results[name] = {"baseline": b, "model": m,
                         "per_class": classification_report(ys, preds, labels=C.CATEGORIES, output_dict=True, zero_division=0),
                         "confusion_matrix": confusion_matrix(ys, preds, labels=C.CATEGORIES).tolist()}

    # 4) latency: single-alert inference, as the Lambda does it
    lat = []
    for t in X_te[:300]:
        s = time.perf_counter()
        model.predict(pipe, t)
        lat.append((time.perf_counter() - s) * 1000)
    lat.sort()
    results["latency_ms"] = {"median": round(statistics.median(lat), 2), "p95": round(lat[int(0.95 * len(lat))], 2)}

    OUT.mkdir(exist_ok=True)
    (OUT / "metrics.json").write_text(json.dumps(results, indent=2))
    lines = [f"# Evaluation results\n", f"Threshold (chosen by leave-templates-out CV on training data): {threshold}\n",
             f"Train/test/novel sizes: {results['sizes']}\n", f"Latency per alert: {results['latency_ms']}\n"]
    for name in ("standard_test", "novel_phrasing"):
        r = results[name]
        lines.append(f"\n## {name}\n\n| Metric | Keyword baseline | AI model |\n|---|---|---|")
        lines.append(f"| Accuracy (top-1) | {r['baseline']['accuracy']:.3f} | {r['model']['accuracy']:.3f} |")
        lines.append(f"| Macro F1 | {r['baseline']['macro_f1']:.3f} | {r['model']['macro_f1']:.3f} |")
        lines.append(f"| Misrouted to wrong team | {r['baseline']['misroute_rate']:.3f} | {r['model']['misroute_rate']:.3f} |")
        lines.append(f"| Unrouted / sent to human triage | {r['baseline']['unrouted_rate']:.3f} | {r['model']['human_triage_rate']:.3f} |")
        lines.append(f"| Auto-routed share | {1 - r['baseline']['unrouted_rate']:.3f} | {r['model']['auto_route_rate']:.3f} |")
    (OUT / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
