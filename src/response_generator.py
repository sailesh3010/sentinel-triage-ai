"""
response_generator.py — Grounded response generation for AppleSupport.

Two modes:
  1. LLM Mode: Uses Groq API (free tier, Llama 3.3 70B) for high-quality generation.
  2. Template Mode: Offline grounded synthesis using retrieved historical responses
     and curated brand templates — no API key required.
"""

import os
import re
import json
import logging
import random
from typing import List, Dict, Optional

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()

# ---------------------------------------------------------------------------
# Brand-aligned response templates (AppleSupport voice)
# ---------------------------------------------------------------------------

TEMPLATES = {
    "software_troubleshooting": {
        "auto_handle": [
            "We understand how frustrating {issue_type} issues can be. {resolution_steps} If you need further help, we're here for you. {dm_link}",
            "Thanks for reaching out about this! {resolution_steps} Let us know if that helps. {dm_link}",
            "We'd love to help sort this out. {resolution_steps} Feel free to DM us if you need more assistance. {dm_link}",
        ],
        "escalate": [
            "We want to make sure we get this resolved for you. Let's continue in DM where we can dig deeper. {dm_link}",
            "Thanks for letting us know about this. Let's take a closer look — DM us your device model and iOS version. {dm_link}",
        ],
    },
    "battery_power": {
        "auto_handle": [
            "We understand battery concerns are important. {resolution_steps} If the issue persists, DM us and we'll help further. {dm_link}",
            "Let's get your battery sorted out. {resolution_steps} Reach out if you need more help. {dm_link}",
        ],
        "escalate": [
            "Battery issues can sometimes need a closer look. DM us your device model and we'll run through some diagnostics together. {dm_link}",
        ],
    },
    "connectivity": {
        "auto_handle": [
            "Connectivity issues can be tricky. {resolution_steps} Let us know how it goes! {dm_link}",
            "We'd like to help get you connected. {resolution_steps} DM us if you need more help. {dm_link}",
        ],
        "escalate": [
            "Let's troubleshoot this connection issue together. DM us your device details and we'll work through it. {dm_link}",
        ],
    },
    "account_security": {
        "escalate": [
            "Your account security is our priority. Let's handle this securely — please DM us so we can verify your identity and assist. {dm_link}",
            "We take account security very seriously. DM us and we'll help get your account sorted out safely. {dm_link}",
        ],
    },
    "hardware_damage": {
        "escalate": [
            "We're sorry to hear about the damage. Let's get this looked at — DM us your device details and location so we can find the best service option. {dm_link}",
            "We'd like to help with this. DM us so we can discuss repair options and check your coverage. {dm_link}",
        ],
    },
    "billing_store": {
        "escalate": [
            "We understand billing concerns are important. DM us your details and we'll look into this for you right away. {dm_link}",
            "Let's get this billing matter sorted out. Please DM us so we can review your account securely. {dm_link}",
        ],
    },
}

DM_LINK = "https://t.co/GDrqU22YpT"


def _extract_resolution_from_historical(retrieved: List[Dict]) -> str:
    """Extract actionable resolution steps from retrieved historical responses."""
    if not retrieved:
        return "Try restarting your device and checking for software updates."

    best = retrieved[0]
    response = best["support_response"]

    # Clean up the historical response
    # Remove @mentions
    cleaned = re.sub(r'@\S+', '', response).strip()
    # Remove generic DM links
    cleaned = re.sub(r'https://t\.co/\S+', '', cleaned).strip()
    # Remove "We'd love to help" generic prefixes (we'll add our own)
    for prefix in ["We'd love to help.", "We'd like to help.",
                   "We're happy to help.", "We can help with that.",
                   "We're here to help."]:
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):].strip()

    if len(cleaned) < 10:
        cleaned = "Try restarting your device and ensuring you're on the latest software version."

    return cleaned


def _synthesize_template_response(
    intent: str,
    escalation_decision: str,
    retrieved: List[Dict],
    customer_text: str,
) -> str:
    """Generate a response using templates + retrieved historical context."""
    decision_key = "auto_handle" if escalation_decision == "AUTO_HANDLE" else "escalate"
    templates = TEMPLATES.get(intent, TEMPLATES["software_troubleshooting"])
    template_list = templates.get(decision_key, templates.get("escalate", []))

    if not template_list:
        template_list = ["We're here to help. DM us your details and we'll assist you. {dm_link}"]

    template = random.choice(template_list)

    # Extract resolution steps from retrieved cases
    resolution_steps = _extract_resolution_from_historical(retrieved)

    # Detect issue type from customer text
    issue_type = "this"
    text_lower = customer_text.lower()
    if "update" in text_lower or "ios" in text_lower or "macos" in text_lower:
        issue_type = "software update"
    elif "battery" in text_lower or "drain" in text_lower:
        issue_type = "battery"
    elif "wifi" in text_lower or "wi-fi" in text_lower or "bluetooth" in text_lower:
        issue_type = "connectivity"
    elif "screen" in text_lower:
        issue_type = "screen"
    elif "apple id" in text_lower or "icloud" in text_lower:
        issue_type = "account"

    response = template.format(
        issue_type=issue_type,
        resolution_steps=resolution_steps,
        dm_link=DM_LINK,
    )

    return response.strip()


# ---------------------------------------------------------------------------
# Groq LLM Response Generator
# ---------------------------------------------------------------------------

def _build_llm_prompt(
    customer_text: str,
    intent: str,
    escalation_decision: str,
    escalation_reason: str,
    retrieved: List[Dict],
) -> str:
    """Build the LLM prompt with retrieved context for grounded generation."""
    context_block = ""
    for i, r in enumerate(retrieved[:3], 1):
        context_block += f"\n  Case {i} (similarity: {r['similarity']:.2f}):\n"
        context_block += f"    Customer: {r['customer_query']}\n"
        context_block += f"    Apple's Response: {r['support_response']}\n"

    prompt = f"""You are an AppleSupport customer service agent on Twitter. Generate a helpful, 
empathetic reply to the customer message below.

STRICT RULES:
- Match Apple's professional, calm, empathetic brand voice.
- NEVER ask for passwords, credit card numbers, or sensitive data in a public tweet.
- If the decision is ESCALATE, guide the customer to DM for secure handling.
- Ground your response in the historical cases provided — do NOT hallucinate information.
- Keep the reply under 280 characters (Twitter limit) when possible.
- Include the DM link {DM_LINK} when directing to DM.

CUSTOMER MESSAGE: {customer_text}

CLASSIFIED INTENT: {intent}
DECISION: {escalation_decision}
REASON: {escalation_reason}

HISTORICAL SIMILAR CASES:{context_block}

Generate a single tweet reply as AppleSupport. No preamble, no quotes — just the tweet text."""

    return prompt


def _call_groq_api(prompt: str) -> Optional[str]:
    """Call Groq API with the given prompt. Returns response text or None."""
    load_dotenv(override=True)
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "your_groq_api_key_here":
        return None

    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        model = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=300,
            top_p=0.9,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"Groq API call failed: {e}")
        return None


class ResponseGenerator:
    """
    Grounded response generator for AppleSupport.
    Uses LLM (Groq) when available, falls back to template synthesis.
    """

    def __init__(self, use_llm: bool = True):
        self.use_llm = use_llm
        self._llm_available = None

    @property
    def llm_available(self) -> bool:
        load_dotenv(override=True)
        api_key = os.getenv("GROQ_API_KEY")
        return bool(api_key and api_key != "your_groq_api_key_here")

    def generate(
        self,
        customer_text: str,
        intent: str,
        escalation_decision: str,
        escalation_reason: str,
        retrieved: List[Dict],
    ) -> Dict:
        """
        Generate a grounded response.

        Returns:
            {
                "response": str,
                "method": "llm" or "template",
                "grounding_sources": list of retrieval IDs used
            }
        """
        grounding_sources = [r["id"] for r in retrieved[:3]]

        # Try LLM first
        if self.use_llm and self.llm_available:
            prompt = _build_llm_prompt(
                customer_text, intent, escalation_decision,
                escalation_reason, retrieved,
            )
            llm_response = _call_groq_api(prompt)
            if llm_response:
                return {
                    "response": llm_response,
                    "method": "llm",
                    "grounding_sources": grounding_sources,
                }

        # Fallback: template-based synthesis
        response = _synthesize_template_response(
            intent, escalation_decision, retrieved, customer_text,
        )
        return {
            "response": response,
            "method": "template",
            "grounding_sources": grounding_sources,
        }


# ---------------------------------------------------------------------------
# Baseline response generators
# ---------------------------------------------------------------------------

class TrivialResponseGenerator:
    """Always returns the same canned response."""

    CANNED = (
        "We're here to help! Please DM us your device model and a description "
        f"of the issue so we can assist you. {DM_LINK}"
    )

    def generate(self, customer_text: str, **kwargs) -> Dict:
        return {
            "response": self.CANNED,
            "method": "canned",
            "grounding_sources": [],
        }


class NearestNeighborResponseGenerator:
    """Returns the verbatim support response from the single nearest neighbor."""

    def generate(
        self,
        customer_text: str,
        retrieved: List[Dict],
        **kwargs,
    ) -> Dict:
        if retrieved:
            return {
                "response": retrieved[0]["support_response"],
                "method": "1nn_verbatim",
                "grounding_sources": [retrieved[0]["id"]],
            }
        return TrivialResponseGenerator().generate(customer_text)
