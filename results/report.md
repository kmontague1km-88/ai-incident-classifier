# Evaluation results

Threshold (chosen by leave-templates-out CV on training data): 0.65

Train/test/novel sizes: {'train': 1470, 'test': 630, 'novel': 420}

Latency per alert: {'median': 2.49, 'p95': 2.81}


## standard_test

| Metric | Keyword baseline | AI model |
|---|---|---|
| Accuracy (top-1) | 0.905 | 1.000 |
| Macro F1 | 0.929 | 1.000 |
| Misrouted to wrong team | 0.041 | 0.000 |
| Unrouted / sent to human triage | 0.054 | 0.000 |
| Auto-routed share | 0.946 | 1.000 |

## novel_phrasing

| Metric | Keyword baseline | AI model |
|---|---|---|
| Accuracy (top-1) | 0.293 | 0.812 |
| Macro F1 | 0.385 | 0.800 |
| Misrouted to wrong team | 0.055 | 0.000 |
| Unrouted / sent to human triage | 0.652 | 0.543 |
| Auto-routed share | 0.348 | 0.457 |
