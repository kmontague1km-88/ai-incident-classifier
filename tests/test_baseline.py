from triage import baseline


def test_keyword_router_matches_known_wording():
    assert baseline.classify("ALARM: CPUUtilization > 95% on web-01") == "COMPUTE_CAPACITY"
    assert baseline.classify("Replica lag on orders-db is 120 seconds") == "DATABASE"
    assert baseline.classify("AWS Budgets: actual spend exceeded 90% of monthly budget") == "COST_BILLING"


def test_keyword_router_misses_new_wording():
    # The weakness the AI enhancement targets: plain-language alerts fall through.
    assert baseline.classify("no space left on device on web-01") == baseline.UNROUTED
    assert baseline.classify("bill jumped $900 overnight, unknown cause") == baseline.UNROUTED


def test_keyword_router_first_match_can_misroute():
    # "public" is a security keyword, so a load balancer alert mentioning a
    # public endpoint is sent to Security instead of Network.
    assert baseline.classify("public load balancer has 0 healthy targets") == "SECURITY"
