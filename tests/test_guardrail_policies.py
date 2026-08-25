from src.guardrails.policies import GuardrailPolicies


def test_default_policies():
    policies = GuardrailPolicies()
    assert policies.max_risk_score == 0.7
    assert policies.min_overall_confidence == 0.7
    assert policies.allowed_query_types == [
        "numeric",
        "comparison",
        "trend",
        "entity",
        "period",
    ]
    assert policies.block_on_numeric_mismatch is False


def test_is_query_allowed():
    policies = GuardrailPolicies()
    assert policies.is_query_allowed("numeric") is True
    # "unknown" is not in allowed_query_types, so it's not allowed
    assert policies.is_query_allowed("unknown") is False


def test_get_risk_level():
    policies = GuardrailPolicies()
    assert policies.get_risk_level(0.1) == "LOW"
    assert policies.get_risk_level(0.6) == "MEDIUM"
    assert policies.get_risk_level(0.9) == "HIGH"