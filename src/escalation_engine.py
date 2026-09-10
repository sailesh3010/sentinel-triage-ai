"""
escalation_engine.py — Multi-factor escalation decision engine.

Decides whether a customer message should be auto-handled or escalated
to a human agent, with a transparent, stated reason for every decision.
"""

import re
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Escalation trigger patterns
# ---------------------------------------------------------------------------

PII_PATTERNS = [
    r"password",
    r"serial\s*number",
    r"credit\s*card",
    r"social\s*security",
    r"ssn",
    r"bank\s*account",
    r"card\s*number",
    r"routing\s*number",
    r"debit\s*card",
]

ACCOUNT_SECURITY_PATTERNS = [
    r"apple\s*id\s*(is\s*)?(locked|disabled|hacked|compromised)",
    r"account\s*(is\s*)?(locked|disabled|hacked|compromised|stolen)",
    r"can'?t\s*(sign|log)\s*in",
    r"two[\s-]*factor",
    r"2fa",
    r"verification\s*code",
    r"activation\s*lock",
    r"find\s*my\s*(iphone|ipad|mac)",
    r"phishing",
    r"unauthorized\s*(access|purchase|charge)",
]

HARDWARE_DAMAGE_PATTERNS = [
    r"(screen|display)\s*(is\s*)?(cracked|shattered|broken|damaged)",
    r"water\s*damage",
    r"(dropped|fell)\s*(in|into)\s*water",
    r"swollen\s*battery",
    r"battery\s*(is\s*)?(swelling|bulging|expanding)",
    r"(button|port|speaker|microphone)\s*(is\s*)?(broken|stuck|not\s*working)",
    r"physical(ly)?\s*damage",
    r"bent\s*(frame|phone|ipad)",
]

FRUSTRATION_PATTERNS = [
    r"(worst|terrible|horrible|disgusting|pathetic|useless)\s*(service|support|company|experience)",
    r"(never|not)\s*(buying|purchasing|using)\s*(apple|iphone|mac|ipad)\s*(again|anymore)",
    r"(switch|switching|moved|moving)\s*to\s*(android|samsung|google|windows)",
    r"(lawyer|lawsuit|sue|legal\s*action|consumer\s*protection|bbb|better\s*business)",
    r"(furious|livid|enraged|outraged|disgusted|appalled)",
    r"(scam|fraud|rip\s*off|ripoff|theft|steal|stealing)",
    r"(complaint|complain)\s*(to|with)\s*(fcc|ftc|attorney\s*general)",
]

BILLING_PATTERNS = [
    r"refund",
    r"charged\s*(me\s*)?(twice|incorrectly|wrongly|without)",
    r"(unexpected|unknown|mysterious)\s*(charge|purchase|transaction)",
    r"(cancel|cancelled|canceling)\s*(my\s*)?(subscription|renewal|order)",
    r"(didn'?t|did\s*not)\s*(authorize|approve)\s*(this\s*)?(purchase|charge|transaction)",
]


def _check_patterns(text: str, patterns: List[str]) -> List[str]:
    """Return list of matched pattern descriptions."""
    matches = []
    for pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            matches.append(pattern)
    return matches


class EscalationEngine:
    """
    Multi-factor escalation decision engine.

    Decision hierarchy (in priority order):
      1. PII exposure → ESCALATE (privacy risk)
      2. Account security → ESCALATE (identity verification needed)
      3. Hardware damage → ESCALATE (physical repair needed)
      4. High frustration → ESCALATE (churn risk)
      5. Billing dispute → ESCALATE (private transaction data)
      6. Low classifier confidence → ESCALATE (uncertainty)
      7. Otherwise → AUTO_HANDLE
    """

    def __init__(self, confidence_threshold: float = 0.35):
        self.confidence_threshold = confidence_threshold

    def decide(
        self,
        text: str,
        intent: str,
        confidence: float,
        retrieval_similarity: float = 1.0,
    ) -> Dict:
        """
        Make an escalation decision.

        Args:
            text: Customer message text
            intent: Predicted intent class
            confidence: Intent classifier confidence score
            retrieval_similarity: Best retrieval match similarity

        Returns:
            {
                "decision": "AUTO_HANDLE" or "ESCALATE",
                "reason": str,
                "factors": list of triggered factors,
                "risk_score": float (0-1, higher = more risky)
            }
        """
        text_lower = text.lower()
        factors = []
        risk_score = 0.0

        # Factor 1: PII exposure
        pii_matches = _check_patterns(text, PII_PATTERNS)
        if pii_matches:
            factors.append("PII_EXPOSURE")
            risk_score += 0.4

        # Factor 2: Account security (regex or classified intent)
        acct_matches = _check_patterns(text, ACCOUNT_SECURITY_PATTERNS)
        if acct_matches or (intent == "account_security" and confidence >= self.confidence_threshold):
            factors.append("ACCOUNT_SECURITY")
            risk_score += 0.35

        # Factor 3: Hardware damage (regex or classified intent)
        hw_matches = _check_patterns(text, HARDWARE_DAMAGE_PATTERNS)
        if hw_matches or (intent == "hardware_damage" and confidence >= self.confidence_threshold):
            factors.append("HARDWARE_DAMAGE")
            risk_score += 0.3

        # Factor 4: Frustration/churn risk
        frust_matches = _check_patterns(text, FRUSTRATION_PATTERNS)
        if frust_matches:
            factors.append("HIGH_FRUSTRATION")
            risk_score += 0.25

        # Factor 5: Billing dispute (regex or classified intent)
        bill_matches = _check_patterns(text, BILLING_PATTERNS)
        if bill_matches or (intent == "billing_store" and confidence >= self.confidence_threshold):
            factors.append("BILLING_DISPUTE")
            risk_score += 0.2

        # Factor 6: Low model confidence
        if confidence < self.confidence_threshold:
            factors.append("LOW_CONFIDENCE")
            risk_score += 0.15

        # Factor 7: Low retrieval similarity (no good precedent found)
        if retrieval_similarity < 0.4:
            factors.append("LOW_RETRIEVAL_MATCH")
            risk_score += 0.1

        # Build decision and reason
        risk_score = min(1.0, risk_score)

        if "PII_EXPOSURE" in factors:
            return {
                "decision": "ESCALATE",
                "reason": "Message contains sensitive personal information (PII). "
                          "Transitioning to secure DM channel to protect customer privacy.",
                "factors": factors,
                "risk_score": round(risk_score, 3),
            }

        if "ACCOUNT_SECURITY" in factors:
            return {
                "decision": "ESCALATE",
                "reason": "Account security issue detected. Requires identity verification "
                          "and private account access via secure DM channel.",
                "factors": factors,
                "risk_score": round(risk_score, 3),
            }

        if "HARDWARE_DAMAGE" in factors:
            return {
                "decision": "ESCALATE",
                "reason": "Physical hardware damage reported. Requires in-person diagnostic "
                          "or Genius Bar appointment for repair assessment.",
                "factors": factors,
                "risk_score": round(risk_score, 3),
            }

        if "HIGH_FRUSTRATION" in factors:
            return {
                "decision": "ESCALATE",
                "reason": "Customer expressing high frustration or churn risk. "
                          "Routing to senior human agent for personalized care.",
                "factors": factors,
                "risk_score": round(risk_score, 3),
            }

        if "BILLING_DISPUTE" in factors:
            return {
                "decision": "ESCALATE",
                "reason": "Billing or transaction dispute detected. Requires access to "
                          "private payment records via secure channel.",
                "factors": factors,
                "risk_score": round(risk_score, 3),
            }

        if "LOW_CONFIDENCE" in factors and "LOW_RETRIEVAL_MATCH" in factors:
            return {
                "decision": "ESCALATE",
                "reason": "Low classification confidence and no strong historical precedent found. "
                          "Flagged for human triage to avoid inaccurate automated guidance.",
                "factors": factors,
                "risk_score": round(risk_score, 3),
            }

        # Default: AUTO_HANDLE
        reason_map = {
            "software_troubleshooting": "Standard software troubleshooting — automated guidance with verified steps available.",
            "battery_power": "Battery/power issue — standard diagnostic steps and guidance available.",
            "connectivity": "Connectivity issue — standard network troubleshooting steps available.",
            "account_security": "Account inquiry — routine guidance available (no security flags detected).",
            "hardware_damage": "Hardware inquiry — standard warranty/service guidance available.",
            "billing_store": "Store/billing inquiry — standard information available.",
        }
        return {
            "decision": "AUTO_HANDLE",
            "reason": reason_map.get(intent, "Standard issue suitable for automated resolution guidance."),
            "factors": factors,
            "risk_score": round(risk_score, 3),
        }


class TrivialEscalationEngine:
    """Baseline 1: Trivial escalation engine (always auto-handles)."""

    def decide(self, text: str, intent: str, confidence: float, retrieval_similarity: float = 1.0) -> Dict:
        return {
            "decision": "AUTO_HANDLE",
            "reason": "Default majority baseline: all incoming inquiries attempted via automated response.",
            "factors": ["BASELINE_DEFAULT"],
            "risk_score": 0.0,
        }


class KeywordEscalationEngine:
    """Baseline 2: Simple keyword-only escalation engine (no model confidence or intent integration)."""

    def decide(self, text: str, intent: str, confidence: float, retrieval_similarity: float = 1.0) -> Dict:
        factors = []
        if _check_patterns(text, PII_PATTERNS):
            factors.append("PII_EXPOSURE")
        if _check_patterns(text, ACCOUNT_SECURITY_PATTERNS):
            factors.append("ACCOUNT_SECURITY")
        if _check_patterns(text, HARDWARE_DAMAGE_PATTERNS):
            factors.append("HARDWARE_DAMAGE")
        if _check_patterns(text, FRUSTRATION_PATTERNS):
            factors.append("HIGH_FRUSTRATION")
        if _check_patterns(text, BILLING_PATTERNS):
            factors.append("BILLING_DISPUTE")

        if factors:
            return {
                "decision": "ESCALATE",
                "reason": f"Keyword trigger detected ({factors[0]}). Escalating to human agent.",
                "factors": factors,
                "risk_score": 0.5,
            }
        return {
            "decision": "AUTO_HANDLE",
            "reason": "No keyword escalation triggers matched.",
            "factors": [],
            "risk_score": 0.1,
        }

