"""
tests/test_pipeline.py — Automated tests for the AI Support Agent pipeline.
"""

import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_intent_taxonomy_complete():
    """Verify all 6 intents are defined with keywords."""
    from src.data_loader import INTENT_TAXONOMY
    assert len(INTENT_TAXONOMY) == 6
    for name, info in INTENT_TAXONOMY.items():
        assert "description" in info
        assert "keywords" in info
        assert len(info["keywords"]) >= 5


def test_heuristic_classification():
    """Test keyword-based intent classification."""
    from src.data_loader import _classify_intent_heuristic
    intent, conf = _classify_intent_heuristic("My iPhone battery drains very fast after update")
    assert intent in ["battery_power", "software_troubleshooting"]
    assert 0 < conf <= 1.0


def test_escalation_pii():
    """PII triggers should always escalate."""
    from src.escalation_engine import EscalationEngine
    engine = EscalationEngine()
    result = engine.decide("My password was stolen please help", "account_security", 0.9)
    assert result["decision"] == "ESCALATE"
    assert "PII" in result["factors"][0] or "ACCOUNT" in result["factors"][0]


def test_escalation_hardware():
    """Hardware damage should escalate."""
    from src.escalation_engine import EscalationEngine
    engine = EscalationEngine()
    result = engine.decide("My screen is cracked and shattered", "hardware_damage", 0.8)
    assert result["decision"] == "ESCALATE"


def test_escalation_auto_handle():
    """Standard software issue with high confidence should auto-handle."""
    from src.escalation_engine import EscalationEngine
    engine = EscalationEngine()
    result = engine.decide("How do I update my iPhone to the latest iOS?", "software_troubleshooting", 0.9)
    assert result["decision"] == "AUTO_HANDLE"


def test_escalation_frustration():
    """Frustrated customer should escalate."""
    from src.escalation_engine import EscalationEngine
    engine = EscalationEngine()
    result = engine.decide("This is the worst service ever I'm switching to Samsung", "software_troubleshooting", 0.9)
    assert result["decision"] == "ESCALATE"
    assert "HIGH_FRUSTRATION" in result["factors"]


def test_trivial_classifier():
    """Trivial classifier always predicts majority class."""
    from src.intent_classifier import TrivialClassifier
    clf = TrivialClassifier()
    result = clf.classify("anything at all")
    assert result["intent"] == "software_troubleshooting"
    assert result["confidence"] == 1.0


def test_trivial_response_generator():
    """Trivial response generator returns canned response."""
    from src.response_generator import TrivialResponseGenerator
    gen = TrivialResponseGenerator()
    result = gen.generate("My phone is broken")
    assert "DM" in result["response"] or "help" in result["response"].lower()
    assert result["method"] == "canned"


def test_template_response_escalate():
    """Template generator produces DM link for escalation."""
    from src.response_generator import ResponseGenerator
    gen = ResponseGenerator(use_llm=False)
    result = gen.generate(
        customer_text="My Apple ID is locked",
        intent="account_security",
        escalation_decision="ESCALATE",
        escalation_reason="Account security issue",
        retrieved=[],
    )
    assert "DM" in result["response"] or "t.co" in result["response"]


def test_golden_set_schema():
    """Verify golden set JSON schema if it exists."""
    path = PROJECT_ROOT / "data" / "golden_set.json"
    if not path.exists():
        return  # Skip if not yet generated

    with open(path, "r", encoding="utf-8") as f:
        golden = json.load(f)

    assert isinstance(golden, list)
    assert len(golden) >= 150  # At least 150 examples

    required_keys = ["id", "customer_tweet", "gold_intent", "gold_escalation",
                     "gold_escalation_reason", "gold_human_reply", "difficulty_tag"]
    for item in golden:
        for key in required_keys:
            assert key in item, f"Missing key {key} in golden set item"
        assert item["gold_intent"] in [
            "software_troubleshooting", "battery_power", "connectivity",
            "account_security", "hardware_damage", "billing_store",
        ]
        assert item["gold_escalation"] in ["AUTO_HANDLE", "ESCALATE"]


def test_evaluator_intent():
    """Test intent evaluation metrics computation."""
    from src.evaluator import evaluate_intent_classification
    preds = ["software_troubleshooting", "battery_power", "connectivity"]
    truth = ["software_troubleshooting", "battery_power", "battery_power"]
    result = evaluate_intent_classification(preds, truth)
    assert 0 <= result["accuracy"] <= 1
    assert "macro_f1" in result
    assert "confusion_matrix" in result


def test_evaluator_escalation():
    """Test escalation evaluation metrics."""
    from src.evaluator import evaluate_escalation
    preds = ["AUTO_HANDLE", "ESCALATE", "AUTO_HANDLE", "ESCALATE"]
    truth = ["AUTO_HANDLE", "ESCALATE", "ESCALATE", "AUTO_HANDLE"]
    result = evaluate_escalation(preds, truth)
    assert 0 <= result["accuracy"] <= 1
    assert 0 <= result["false_auto_handle_rate"] <= 1
    assert result["false_auto_handle_count"] == 1  # 3rd example


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
