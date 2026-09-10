"""
evaluator.py — Automated evaluation metrics for the AI support agent.

Computes:
  - Intent classification: Accuracy, Macro F1, Per-class F1, Confusion Matrix
  - Escalation: Accuracy, Precision, Recall, F1, False Auto-handle Rate
  - Response quality: ROUGE-1/2/L, BLEU, Semantic similarity
"""

import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


# ---------------------------------------------------------------------------
# ROUGE / BLEU computation
# ---------------------------------------------------------------------------

def _compute_rouge(hypothesis: str, reference: str) -> Dict[str, float]:
    """Compute ROUGE-1, ROUGE-2, ROUGE-L F1 scores."""
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        scores = scorer.score(reference, hypothesis)
        return {
            "rouge1": round(scores["rouge1"].fmeasure, 4),
            "rouge2": round(scores["rouge2"].fmeasure, 4),
            "rougeL": round(scores["rougeL"].fmeasure, 4),
        }
    except ImportError:
        logger.warning("rouge-score not installed, skipping ROUGE.")
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}


def _compute_bleu(hypothesis: str, reference: str) -> float:
    """Compute BLEU-4 score (simplified)."""
    from collections import Counter

    hyp_tokens = hypothesis.lower().split()
    ref_tokens = reference.lower().split()

    if len(hyp_tokens) == 0 or len(ref_tokens) == 0:
        return 0.0

    # Unigram precision as simplified BLEU
    hyp_counts = Counter(hyp_tokens)
    ref_counts = Counter(ref_tokens)

    clipped = sum(min(hyp_counts[w], ref_counts.get(w, 0)) for w in hyp_counts)
    precision = clipped / len(hyp_tokens) if len(hyp_tokens) > 0 else 0.0

    # Brevity penalty
    bp = min(1.0, len(hyp_tokens) / len(ref_tokens)) if len(ref_tokens) > 0 else 0.0

    return round(bp * precision, 4)


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate_intent_classification(
    predictions: List[str],
    ground_truth: List[str],
    label_names: Optional[List[str]] = None,
) -> Dict:
    """Evaluate intent classification performance."""
    if label_names is None:
        label_names = sorted(set(ground_truth))

    acc = accuracy_score(ground_truth, predictions)
    precision, recall, f1, support = precision_recall_fscore_support(
        ground_truth, predictions, labels=label_names, average=None, zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        ground_truth, predictions, average="macro", zero_division=0
    )

    cm = confusion_matrix(ground_truth, predictions, labels=label_names)

    per_class = {}
    for i, label in enumerate(label_names):
        per_class[label] = {
            "precision": round(float(precision[i]), 4),
            "recall": round(float(recall[i]), 4),
            "f1": round(float(f1[i]), 4),
            "support": int(support[i]),
        }

    return {
        "accuracy": round(float(acc), 4),
        "macro_precision": round(float(macro_p), 4),
        "macro_recall": round(float(macro_r), 4),
        "macro_f1": round(float(macro_f1), 4),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "label_names": label_names,
    }


def evaluate_escalation(
    predictions: List[str],
    ground_truth: List[str],
) -> Dict:
    """
    Evaluate escalation decision performance.
    Key metric: False Auto-handle Rate = fraction of ESCALATE examples
    that were incorrectly predicted as AUTO_HANDLE.
    """
    acc = accuracy_score(ground_truth, predictions)

    # Binary: ESCALATE = positive, AUTO_HANDLE = negative
    y_true_bin = [1 if g == "ESCALATE" else 0 for g in ground_truth]
    y_pred_bin = [1 if p == "ESCALATE" else 0 for p in predictions]

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_bin, y_pred_bin, average="binary", zero_division=0
    )

    # False Auto-handle Rate: predicted AUTO_HANDLE when truth is ESCALATE
    false_auto_handle = sum(
        1 for gt, pred in zip(ground_truth, predictions)
        if gt == "ESCALATE" and pred == "AUTO_HANDLE"
    )
    total_escalate = sum(1 for g in ground_truth if g == "ESCALATE")
    false_auto_handle_rate = false_auto_handle / total_escalate if total_escalate > 0 else 0.0

    # False Escalation Rate: predicted ESCALATE when truth is AUTO_HANDLE
    false_escalation = sum(
        1 for gt, pred in zip(ground_truth, predictions)
        if gt == "AUTO_HANDLE" and pred == "ESCALATE"
    )
    total_auto = sum(1 for g in ground_truth if g == "AUTO_HANDLE")
    false_escalation_rate = false_escalation / total_auto if total_auto > 0 else 0.0

    return {
        "accuracy": round(float(acc), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "false_auto_handle_rate": round(float(false_auto_handle_rate), 4),
        "false_escalation_rate": round(float(false_escalation_rate), 4),
        "false_auto_handle_count": false_auto_handle,
        "false_escalation_count": false_escalation,
    }


def evaluate_response_quality(
    generated_responses: List[str],
    reference_responses: List[str],
    encoder=None,
) -> Dict:
    """
    Evaluate generated response quality vs human reference responses.
    Metrics: ROUGE-1/2/L, BLEU, Semantic similarity.
    """
    rouge_scores = {"rouge1": [], "rouge2": [], "rougeL": []}
    bleu_scores = []

    for gen, ref in zip(generated_responses, reference_responses):
        rouge = _compute_rouge(gen, ref)
        for k in rouge_scores:
            rouge_scores[k].append(rouge[k])
        bleu_scores.append(_compute_bleu(gen, ref))

    results = {
        "rouge1_mean": round(float(np.mean(rouge_scores["rouge1"])), 4),
        "rouge2_mean": round(float(np.mean(rouge_scores["rouge2"])), 4),
        "rougeL_mean": round(float(np.mean(rouge_scores["rougeL"])), 4),
        "bleu_mean": round(float(np.mean(bleu_scores)), 4),
    }

    # Semantic similarity using sentence-transformers
    if encoder is not None:
        gen_embs = encoder.encode_batch(generated_responses)
        ref_embs = encoder.encode_batch(reference_responses)
        from sklearn.metrics.pairwise import cosine_similarity
        sims = [
            float(cosine_similarity(g.reshape(1, -1), r.reshape(1, -1))[0][0])
            for g, r in zip(gen_embs, ref_embs)
        ]
        results["semantic_similarity_mean"] = round(float(np.mean(sims)), 4)

    return results


def run_full_evaluation(
    agent,
    golden_set: List[Dict],
    encoder=None,
) -> Dict:
    """
    Run full evaluation of an agent against the golden set.
    Returns comprehensive metrics dictionary.
    """
    logger.info(f"Evaluating agent: {agent.agent_name} on {len(golden_set)} examples...")

    predictions = []
    for item in golden_set:
        pred = agent.predict(item["customer_tweet"])
        predictions.append(pred)

    # Intent classification
    pred_intents = [p["intent"] for p in predictions]
    gold_intents = [g["gold_intent"] for g in golden_set]
    intent_metrics = evaluate_intent_classification(pred_intents, gold_intents)

    # Escalation
    pred_escalations = [p["escalation_decision"] for p in predictions]
    gold_escalations = [g["gold_escalation"] for g in golden_set]
    escalation_metrics = evaluate_escalation(pred_escalations, gold_escalations)

    # Response quality
    gen_responses = [p["response"] for p in predictions]
    ref_responses = [g["gold_human_reply"] for g in golden_set]
    response_metrics = evaluate_response_quality(gen_responses, ref_responses, encoder=encoder)

    results = {
        "agent_name": agent.agent_name,
        "num_examples": len(golden_set),
        "intent": intent_metrics,
        "escalation": escalation_metrics,
        "response": response_metrics,
        "predictions": predictions,
    }

    return results


def save_results(results: Dict, filename: str = None):
    """Save evaluation results to JSON."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if filename is None:
        filename = f"{results['agent_name']}_results.json"
    path = RESULTS_DIR / filename

    # Remove predictions from saved file (too large)
    save_data = {k: v for k, v in results.items() if k != "predictions"}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved to {path}")
    return path


def print_results_table(all_results: List[Dict]):
    """Print a comparison table of all agents."""
    print("\n" + "=" * 100)
    print("EVALUATION RESULTS COMPARISON")
    print("=" * 100)

    header = f"{'Metric':<35} "
    for r in all_results:
        header += f"| {r['agent_name']:>20} "
    print(header)
    print("-" * len(header))

    metrics = [
        ("Intent Accuracy", lambda r: r["intent"]["accuracy"]),
        ("Intent Macro F1", lambda r: r["intent"]["macro_f1"]),
        ("Escalation Accuracy", lambda r: r["escalation"]["accuracy"]),
        ("Escalation F1", lambda r: r["escalation"]["f1"]),
        ("False Auto-handle Rate ↓", lambda r: r["escalation"]["false_auto_handle_rate"]),
        ("ROUGE-1", lambda r: r["response"]["rouge1_mean"]),
        ("ROUGE-L", lambda r: r["response"]["rougeL_mean"]),
        ("BLEU", lambda r: r["response"]["bleu_mean"]),
    ]

    if "semantic_similarity_mean" in all_results[0].get("response", {}):
        metrics.append(("Semantic Similarity", lambda r: r["response"].get("semantic_similarity_mean", "N/A")))

    for name, getter in metrics:
        row = f"{name:<35} "
        for r in all_results:
            val = getter(r)
            if isinstance(val, float):
                row += f"| {val:>20.4f} "
            else:
                row += f"| {str(val):>20} "
        print(row)

    print("=" * 100)
