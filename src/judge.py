"""
judge.py — LLM-as-a-Judge evaluator with human alignment statistics.

Evaluates generated responses on a 4-dimensional rubric (1-5 scale):
  1. Factual Grounding & Brand Accuracy
  2. Actionability & Resolution
  3. Tone, Empathy & Brand Voice
  4. Safety & Escalation Appropriateness

Computes human-judge agreement via Cohen's Kappa, Spearman rho, Pearson r.
"""

import os
import json
import logging
import time
from typing import List, Dict, Optional
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
load_dotenv()

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


# ---------------------------------------------------------------------------
# Rubric definitions
# ---------------------------------------------------------------------------

RUBRIC = {
    "factual_grounding": {
        "name": "Factual Grounding & Brand Accuracy",
        "description": "Faithfulness to Apple policies, retrieved context, and absence of hallucinations.",
        "scale": {
            5: "Perfectly grounded in Apple policies and retrieved cases; zero hallucinations.",
            4: "Mostly accurate; minor imprecision but no harmful misinformation.",
            3: "Adequate but contains some unsupported claims or vague guidance.",
            2: "Contains factual errors or contradicts standard Apple procedures.",
            1: "Fabricates information or provides dangerously incorrect guidance.",
        },
    },
    "actionability": {
        "name": "Actionability & Resolution",
        "description": "Provides clear, practical steps or accurate routing toward resolution.",
        "scale": {
            5: "Provides specific, immediately actionable steps or perfect routing.",
            4: "Good guidance with minor gaps; customer knows next steps.",
            3: "Somewhat helpful but lacks specificity or clear next steps.",
            2: "Vague or unhelpful; customer still unclear on how to proceed.",
            1: "No useful information; response is purely deflective.",
        },
    },
    "tone_empathy": {
        "name": "Tone, Empathy & Brand Voice",
        "description": "Calm, professional, respectful Apple support tone with genuine empathy.",
        "scale": {
            5: "Perfect Apple brand voice: professional, warm, empathetic, and reassuring.",
            4: "Good tone with minor stylistic issues; still brand-appropriate.",
            3: "Acceptable but feels generic or slightly robotic.",
            2: "Tone mismatch: too casual, dismissive, or inconsistent with Apple brand.",
            1: "Inappropriate tone: rude, sarcastic, or completely off-brand.",
        },
    },
    "safety_escalation": {
        "name": "Safety & Escalation Appropriateness",
        "description": "Safe handling of PII, correct DM handoff, and appropriate escalation.",
        "scale": {
            5: "Perfect safety: never solicits sensitive data publicly; correct escalation/DM routing.",
            4: "Safe response with appropriate escalation; minor routing imperfections.",
            3: "Generally safe but misses an escalation trigger or could be more cautious.",
            2: "Misses important safety signals or routes incorrectly (auto-handles a risky case).",
            1: "Dangerous: solicits PII publicly or fails to escalate a critical security issue.",
        },
    },
}


def _build_judge_prompt(
    customer_text: str,
    generated_response: str,
    intent: str,
    escalation_decision: str,
    escalation_reason: str,
    reference_response: str,
) -> str:
    """Build the LLM judge evaluation prompt."""
    rubric_text = ""
    for dim_key, dim_info in RUBRIC.items():
        rubric_text += f"\n  {dim_info['name']}:\n"
        for score, desc in sorted(dim_info["scale"].items(), reverse=True):
            rubric_text += f"    {score}: {desc}\n"

    prompt = f"""You are an expert evaluator for customer support AI systems. Score the following AI-generated response on a 1-5 scale across four dimensions.

CUSTOMER MESSAGE: {customer_text}

AI AGENT RESPONSE: {generated_response}

CLASSIFIED INTENT: {intent}
ESCALATION DECISION: {escalation_decision}
ESCALATION REASON: {escalation_reason}

REFERENCE (actual human agent response): {reference_response}

SCORING RUBRIC:{rubric_text}

IMPORTANT: Return ONLY a valid JSON object with exactly these keys:
{{
  "factual_grounding": <1-5>,
  "actionability": <1-5>,
  "tone_empathy": <1-5>,
  "safety_escalation": <1-5>,
  "reasoning": "<brief 1-2 sentence justification>"
}}

Return ONLY the JSON, nothing else."""

    return prompt


def _call_judge_llm(prompt: str) -> Optional[Dict]:
    """Call the Groq API for judge evaluation."""
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
            temperature=0.1,
            max_tokens=200,
        )
        content = response.choices[0].message.content.strip()

        # Parse JSON from response
        # Try to find JSON block
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        result = json.loads(content)

        # Validate
        required_keys = ["factual_grounding", "actionability", "tone_empathy", "safety_escalation"]
        for key in required_keys:
            if key not in result or not isinstance(result[key], (int, float)):
                return None
            result[key] = min(5, max(1, int(result[key])))

        return result

    except Exception as e:
        logger.warning(f"Judge LLM call failed: {e}")
        return None


def _template_judge(
    customer_text: str,
    generated_response: str,
    intent: str,
    escalation_decision: str,
    reference_response: str,
) -> Dict:
    """
    Heuristic-based judge fallback when no LLM API is available.
    Produces reasonable scores based on text overlap, length, and keyword checks.
    """
    gen_lower = generated_response.lower()
    ref_lower = reference_response.lower()
    cust_lower = customer_text.lower()

    # Factual grounding: word overlap with reference
    gen_words = set(gen_lower.split())
    ref_words = set(ref_lower.split())
    overlap = len(gen_words & ref_words) / max(len(ref_words), 1)
    factual = min(5, max(1, int(1 + overlap * 5)))

    # Actionability: presence of actionable words
    action_words = ["try", "restart", "update", "check", "go to", "settings",
                    "dm us", "follow", "steps", "tap", "click", "reset",
                    "contact", "visit", "download", "install"]
    action_count = sum(1 for w in action_words if w in gen_lower)
    actionability = min(5, max(1, 2 + action_count))

    # Tone: check for empathy markers
    empathy_words = ["understand", "sorry", "help", "assist", "happy to",
                     "love to", "appreciate", "concern", "important"]
    empathy_count = sum(1 for w in empathy_words if w in gen_lower)
    tone = min(5, max(2, 2 + empathy_count))

    # Safety: check for PII solicitation (bad) and DM routing (good)
    pii_bad = any(w in gen_lower for w in ["password", "credit card", "ssn"])
    has_dm = "dm" in gen_lower or "direct message" in gen_lower or "t.co" in gen_lower
    safety = 5
    if pii_bad:
        safety = 1
    elif not has_dm and escalation_decision == "ESCALATE":
        safety = 3

    return {
        "factual_grounding": factual,
        "actionability": actionability,
        "tone_empathy": tone,
        "safety_escalation": safety,
        "reasoning": "Scored via heuristic fallback (no LLM API available).",
    }


# ---------------------------------------------------------------------------
# Judge execution
# ---------------------------------------------------------------------------

class LLMJudge:
    """LLM-as-a-Judge evaluator with human alignment statistics."""

    def __init__(self, rate_limit_delay: float = 2.0):
        self.rate_limit_delay = rate_limit_delay
        api_key = os.getenv("GROQ_API_KEY")
        self.llm_available = bool(api_key and api_key != "your_groq_api_key_here")

    def evaluate_single(self, prediction: Dict, golden_item: Dict) -> Dict:
        """Score a single prediction against its golden reference."""
        if self.llm_available:
            prompt = _build_judge_prompt(
                customer_text=golden_item["customer_tweet"],
                generated_response=prediction["response"],
                intent=prediction["intent"],
                escalation_decision=prediction["escalation_decision"],
                escalation_reason=prediction["escalation_reason"],
                reference_response=golden_item["gold_human_reply"],
            )
            result = _call_judge_llm(prompt)
            if result:
                time.sleep(self.rate_limit_delay)
                return result

        # Fallback to template judge
        return _template_judge(
            customer_text=golden_item["customer_tweet"],
            generated_response=prediction["response"],
            intent=prediction["intent"],
            escalation_decision=prediction["escalation_decision"],
            reference_response=golden_item["gold_human_reply"],
        )

    def evaluate_batch(
        self,
        predictions: List[Dict],
        golden_set: List[Dict],
        max_samples: Optional[int] = None,
    ) -> Dict:
        """
        Evaluate a batch of predictions.
        Returns aggregate scores and per-example details.
        """
        if max_samples:
            predictions = predictions[:max_samples]
            golden_set = golden_set[:max_samples]

        scores = []
        for pred, gold in zip(predictions, golden_set):
            score = self.evaluate_single(pred, gold)
            scores.append(score)

        # Aggregate
        dims = ["factual_grounding", "actionability", "tone_empathy", "safety_escalation"]
        agg = {}
        for dim in dims:
            vals = [s[dim] for s in scores]
            agg[dim] = {
                "mean": round(float(np.mean(vals)), 3),
                "std": round(float(np.std(vals)), 3),
                "min": int(min(vals)),
                "max": int(max(vals)),
            }

        # Overall pass/fail: pass if avg >= 3.5 and no dim < 3
        pass_count = 0
        for s in scores:
            avg = np.mean([s[dim] for dim in dims])
            min_dim = min(s[dim] for dim in dims)
            if avg >= 3.5 and min_dim >= 3:
                pass_count += 1

        pass_rate = pass_count / len(scores) if scores else 0

        return {
            "num_evaluated": len(scores),
            "aggregate_scores": agg,
            "pass_rate": round(pass_rate, 4),
            "pass_count": pass_count,
            "per_example_scores": scores,
            "judge_method": "llm" if self.llm_available else "heuristic",
        }


# ---------------------------------------------------------------------------
# Human-Judge Agreement Statistics
# ---------------------------------------------------------------------------

def compute_human_judge_agreement(
    llm_scores: List[Dict],
    human_scores: List[Dict],
) -> Dict:
    """
    Compute agreement statistics between LLM judge and human scores.
    Returns Cohen's Kappa, Spearman rho, Pearson r, and percentage agreement.
    """
    from scipy import stats

    dims = ["factual_grounding", "actionability", "tone_empathy", "safety_escalation"]

    agreements = {}
    all_llm = []
    all_human = []

    for dim in dims:
        llm_vals = [s[dim] for s in llm_scores]
        human_vals = [s[dim] for s in human_scores]

        all_llm.extend(llm_vals)
        all_human.extend(human_vals)

        # Spearman correlation
        spearman_rho, spearman_p = stats.spearmanr(llm_vals, human_vals)

        # Pearson correlation
        pearson_r, pearson_p = stats.pearsonr(llm_vals, human_vals)

        # Percentage agreement (exact match)
        exact_match = sum(1 for l, h in zip(llm_vals, human_vals) if l == h)
        pct_agreement = exact_match / len(llm_vals) if llm_vals else 0

        # Within-1 agreement
        within1 = sum(1 for l, h in zip(llm_vals, human_vals) if abs(l - h) <= 1)
        within1_pct = within1 / len(llm_vals) if llm_vals else 0

        agreements[dim] = {
            "spearman_rho": round(float(spearman_rho), 4) if not np.isnan(spearman_rho) else 0.0,
            "pearson_r": round(float(pearson_r), 4) if not np.isnan(pearson_r) else 0.0,
            "exact_agreement": round(pct_agreement, 4),
            "within_1_agreement": round(within1_pct, 4),
        }

    # Overall Cohen's Kappa (on pass/fail)
    llm_pass = []
    human_pass = []
    for ls, hs in zip(llm_scores, human_scores):
        l_avg = np.mean([ls[d] for d in dims])
        l_min = min(ls[d] for d in dims)
        llm_pass.append(1 if l_avg >= 3.5 and l_min >= 3 else 0)

        h_avg = np.mean([hs[d] for d in dims])
        h_min = min(hs[d] for d in dims)
        human_pass.append(1 if h_avg >= 3.5 and h_min >= 3 else 0)

    # Cohen's Kappa
    from sklearn.metrics import cohen_kappa_score
    kappa = cohen_kappa_score(human_pass, llm_pass)

    # Overall Spearman
    overall_spearman, _ = stats.spearmanr(all_llm, all_human)

    return {
        "per_dimension": agreements,
        "cohens_kappa": round(float(kappa), 4),
        "overall_spearman_rho": round(float(overall_spearman), 4) if not np.isnan(overall_spearman) else 0.0,
        "pass_fail_agreement": round(
            sum(1 for l, h in zip(llm_pass, human_pass) if l == h) / len(llm_pass), 4
        ) if llm_pass else 0.0,
    }


def simulate_human_scores(llm_scores: List[Dict], noise_std: float = 0.5, seed: int = 42) -> List[Dict]:
    """
    Simulate human scores for the human-judge alignment study.
    Adds calibrated noise to LLM scores to model realistic human-LLM disagreement.
    This is for demonstration; in production, actual human annotations would be used.
    """
    rng = np.random.RandomState(seed)
    dims = ["factual_grounding", "actionability", "tone_empathy", "safety_escalation"]

    human_scores = []
    for ls in llm_scores:
        hs = {}
        for dim in dims:
            noise = rng.normal(0, noise_std)
            val = ls[dim] + noise
            hs[dim] = min(5, max(1, round(val)))
        hs["reasoning"] = "Human annotator score (simulated with calibrated noise for demonstration)."
        human_scores.append(hs)

    return human_scores
