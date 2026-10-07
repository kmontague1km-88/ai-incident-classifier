"""Synthetic, labeled alert dataset.

Real alert text from a production account was not available, so alerts are
generated from templates modeled on CloudWatch alarm reasons, GuardDuty
finding titles, AWS Budgets notifications, and application error messages.
Two template sets are kept apart on purpose:

* TRAIN_TEMPLATES feed the training split and the standard test split.
* NOVEL_TEMPLATES use different wording (paraphrases, vendor jargon, terse
  pager-style text) and are NEVER used in training. They measure how the
  model handles phrasing it has not seen, which is closer to real life.
"""
import csv
import random
from pathlib import Path

HOSTS = ["web-01", "web-07", "api-prod-3", "batch-worker-2", "i-0a1b2c3d4e5f", "i-09f8e7d6c5b4",
         "ip-10-0-3-17", "orders-svc", "payments-api", "reporting-node"]
DBS = ["orders-db", "rds-prod-main", "customers-aurora", "inventory-ddb", "analytics-pg"]
REGIONS = ["us-east-1", "us-west-2", "eu-west-1"]
BUCKETS = ["prod-logs", "customer-uploads", "backup-archive", "data-lake-raw"]
USERS = ["svc-deploy", "jdoe", "admin-temp", "ci-runner", "contractor-07"]
IPS = ["203.0.113.45", "198.51.100.23", "192.0.2.88", "185.220.101.4"]
SERVICES = ["checkout", "search", "auth", "catalog", "notifications"]


def _fill(t, rng):
    return t.format(
        host=rng.choice(HOSTS), db=rng.choice(DBS), region=rng.choice(REGIONS),
        bucket=rng.choice(BUCKETS), user=rng.choice(USERS), ip=rng.choice(IPS),
        svc=rng.choice(SERVICES), pct=rng.randint(80, 99), n=rng.randint(3, 500),
        ms=rng.randint(400, 9000), amt=rng.randint(120, 9800), ver=f"v{rng.randint(1, 9)}.{rng.randint(0, 30)}")


TRAIN_TEMPLATES = {
    "COMPUTE_CAPACITY": [
        "ALARM: CPUUtilization > {pct}% for 15 minutes on {host}",
        "High CPU usage detected on instance {host} ({pct}%)",
        "MemoryUtilization above threshold on {host}: {pct}% used",
        "Auto Scaling group could not launch instances: InsufficientInstanceCapacity in {region}",
        "EC2 status check failed for {host}; instance unreachable",
        "Instance {host} out of memory, OOM killer terminated process",
        "Load average on {host} is {n}, exceeds number of vCPUs",
        "ECS service {svc} cannot place tasks: insufficient CPU in cluster",
        "Lambda function {svc} throttled: concurrency limit reached",
        "Kubernetes node {host} NotReady, pods evicted due to memory pressure",
    ],
    "NETWORK": [
        "ALARM: TargetResponseTime > {ms} ms on application load balancer",
        "Packet loss of {pct}% between {region} and on-premises data center",
        "VPN tunnel to corporate network is DOWN",
        "Route 53 health check failed for {svc}.example.com",
        "DNS resolution failures for {svc} internal endpoint",
        "NAT gateway ErrorPortAllocation count {n} in {region}",
        "Elevated network latency to {host}: {ms} ms round trip",
        "Load balancer has 0 healthy targets in target group {svc}-tg",
        "Direct Connect connection state changed to down",
        "TLS handshake timeouts between API gateway and {svc} backend",
    ],
    "DATABASE": [
        "ALARM: DatabaseConnections > {n} on {db}",
        "Replica lag on {db} is {n} seconds",
        "Deadlock detected in {db} on table orders",
        "RDS instance {db} storage-full event, writes failing",
        "DynamoDB table {db} ProvisionedThroughputExceededException on writes",
        "Slow query log: {n} queries over {ms} ms on {db}",
        "Aurora failover initiated for cluster {db}",
        "Too many connections error from application to {db}",
        "Database CPU at {pct}% on {db} due to long running query",
        "Backup job for {db} failed: snapshot quota exceeded",
    ],
    "STORAGE": [
        "ALARM: disk usage {pct}% on /var on {host}",
        "EBS volume attached to {host} BurstBalance below 10%",
        "Filesystem / is {pct}% full on {host}",
        "S3 bucket {bucket} replication failed for {n} objects",
        "EBS volume queue length high, IOPS limit reached on {host}",
        "Log partition almost full on {host}, {pct}% used",
        "EFS file system PercentIOLimit at {pct}%",
        "Snapshot creation failed: volume {host} in error state",
        "S3 PUT requests returning SlowDown on bucket {bucket}",
        "Inode usage {pct}% on data volume of {host}",
    ],
    "SECURITY": [
        "GuardDuty finding: UnauthorizedAccess:EC2/SSHBruteForce against {host} from {ip}",
        "{n} failed console login attempts for user {user} from {ip}",
        "IAM policy AdministratorAccess attached to user {user} outside change window",
        "CloudTrail logging stopped by {user}",
        "S3 bucket {bucket} made public by {user}",
        "WAF blocked {n} SQL injection attempts against {svc}",
        "GuardDuty: CryptoCurrency mining activity detected on {host}",
        "Access key for {user} used from unusual location {ip}",
        "Security group opened port 22 to 0.0.0.0/0 by {user}",
        "Root account login detected from {ip}",
    ],
    "APPLICATION_DEPLOY": [
        "ALARM: HTTPCode_Target_5XX_Count > {n} for {svc}",
        "Deployment {ver} of {svc} failed health checks, rollback started",
        "CodePipeline stage Deploy failed for {svc}",
        "Unhandled exception rate increased for {svc} after release {ver}",
        "Application {svc} returning 503 Service Unavailable",
        "Container for {svc} in CrashLoopBackOff",
        "Error rate for {svc} API at {pct}% of requests",
        "Canary deployment of {svc} {ver} exceeded error budget",
        "NullPointerException spike in {svc} logs",
        "Synthetic canary for {svc} checkout flow failing",
    ],
    "COST_BILLING": [
        "AWS Budgets: actual spend exceeded {pct}% of monthly budget",
        "Cost anomaly detected: EC2 spend up ${amt} today in {region}",
        "Forecasted monthly cost exceeds budget by ${amt}",
        "Unexpected NAT gateway data processing charges of ${amt}",
        "Savings Plan utilization dropped to {pct}%",
        "Untagged resources accruing ${amt} per day",
        "Cost Explorer: S3 storage cost increase of ${amt} this week",
        "Reserved instance coverage fell below {pct}% target",
        "Billing alarm: EstimatedCharges > ${amt}",
        "Data transfer out charges spiked by ${amt} in {region}",
    ],
}

NOVEL_TEMPLATES = {
    "COMPUTE_CAPACITY": [
        "{host} pegged, all cores busy, requests queuing",
        "scale-out blocked: no spare capacity for requested instance type",
        "box {host} swapping heavily, RAM exhausted",
        "fleet saturated during peak, need more nodes",
    ],
    "NETWORK": [
        "users report site slow to load, ALB p99 {ms}ms",
        "intermittent connectivity drops to {region}",
        "tunnel flapping between office and VPC",
        "name lookups timing out for internal services",
    ],
    "DATABASE": [
        "{db} primary unhealthy, standby promoted",
        "connection pool exhausted talking to {db}",
        "read replica {db} falling behind master",
        "lock waits piling up on {db}",
    ],
    "STORAGE": [
        "no space left on device on {host}",
        "volume nearly full, logs cannot be written on {host}",
        "throughput credits depleted on gp2 volume of {host}",
        "objects failing to copy to DR bucket {bucket}",
    ],
    "SECURITY": [
        "suspicious API calls by {user} from Tor exit node {ip}",
        "someone disabled audit trail in {region}",
        "brute force against SSH on {host}",
        "new admin rights granted to {user} without ticket",
    ],
    "APPLICATION_DEPLOY": [
        "{svc} throwing 500s since the {ver} rollout",
        "release {ver} broke login for {svc}",
        "pods for {svc} keep restarting after deploy",
        "customers getting errors at checkout on {svc}",
    ],
    "COST_BILLING": [
        "spend tracking way over plan this month (+${amt})",
        "bill jumped ${amt} overnight, unknown cause",
        "budget threshold breached for {region} account",
        "commitment discount coverage slipping to {pct}%",
    ],
}


def generate(templates, per_template, seed):
    rng = random.Random(seed)  # nosec B311 - seeded RNG for reproducible test data, not security
    rows = []
    for label, tlist in templates.items():
        for ti, t in enumerate(tlist):
            for _ in range(per_template):
                text = _fill(t, rng)
                # Light noise: random casing and pager-style prefixes.
                if rng.random() < 0.2:
                    text = text.lower()
                if rng.random() < 0.15:
                    text = rng.choice(["[PagerDuty] ", "[P2] ", "FYI: ", "URGENT "]) + text
                rows.append((text, label, f"{label}-{ti}"))
    rng.shuffle(rows)
    return rows


def build(out_dir=Path(__file__).resolve().parent.parent / "data", seed=42):
    out_dir.mkdir(parents=True, exist_ok=True)
    main = generate(TRAIN_TEMPLATES, per_template=30, seed=seed)       # 2,100 alerts
    novel = generate(NOVEL_TEMPLATES, per_template=15, seed=seed + 1)  # 420 alerts
    for name, rows in (("alerts.csv", main), ("alerts_novel_phrasing.csv", novel)):
        with open(out_dir / name, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["text", "label", "template_id"])
            w.writerows(rows)
    return main, novel


def load(path):
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return [r["text"] for r in rows], [r["label"] for r in rows]


if __name__ == "__main__":
    m, n = build()
    print(f"wrote {len(m)} training-pool alerts and {len(n)} novel-phrasing alerts")
