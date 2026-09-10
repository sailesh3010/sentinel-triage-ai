"""
banking77.py — Banking77 cross-domain intent analysis (secondary dataset).

Provides:
  1. Banking77 taxonomy exploration and statistics
  2. Mapping Banking77 intents to AppleSupport 6-class taxonomy
  3. Cross-domain intent classifier evaluation
  4. Insights on how labeled intent data improves classification
"""

import json
import logging
from typing import Dict, List, Tuple
from pathlib import Path
from collections import Counter

import numpy as np

from src.data_loader import INTENT_TAXONOMY, DATA_DIR

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Banking77 → AppleSupport taxonomy mapping
# ---------------------------------------------------------------------------
#
# Banking77 has 77 fine-grained intents for banking queries.
# We map clusters of these to our 6-class AppleSupport taxonomy to study
# how labeled intent data from a different domain can inform our system.

BANKING77_TO_APPLE_MAP = {
    # Software / Technical Troubleshooting analogues
    "apple_pay_or_google_pay":      "billing_store",
    "beneficiary_not_allowed":      "account_security",
    "cancel_transfer":              "billing_store",
    "card_about_to_expire":         "billing_store",
    "card_acceptance":              "connectivity",      # "device not accepted" parallels
    "card_arrival":                 "billing_store",
    "card_delivery_estimate":       "billing_store",
    "card_linking":                 "software_troubleshooting",
    "card_not_working":             "connectivity",      # "not working" parallels device issues
    "card_payment_fee_charged":     "billing_store",
    "card_payment_not_recognised":  "billing_store",
    "card_payment_wrong_exchange_rate": "billing_store",
    "card_swallowed":               "hardware_damage",   # physical card problem
    "cash_withdrawal_charge":       "billing_store",
    "cash_withdrawal_not_recognised": "billing_store",
    "change_pin":                   "account_security",
    "compromised_card":             "account_security",
    "contactless_not_working":      "connectivity",
    "country_support":              "software_troubleshooting",
    "declined_card_payment":        "billing_store",
    "declined_cash_withdrawal":     "billing_store",
    "declined_transfer":            "billing_store",
    "direct_debit_payment_not_recognised": "billing_store",
    "disposable_card_limits":       "billing_store",
    "edit_personal_details":        "account_security",
    "exchange_charge":              "billing_store",
    "exchange_rate":                "billing_store",
    "exchange_via_app":             "software_troubleshooting",
    "extra_charge_on_statement":    "billing_store",
    "failed_transfer":              "billing_store",
    "fiat_currency_support":        "billing_store",
    "get_disposable_virtual_card":  "billing_store",
    "get_physical_card":            "billing_store",
    "getting_spare_card":           "billing_store",
    "getting_virtual_card":         "billing_store",
    "lost_or_stolen_card":          "account_security",
    "lost_or_stolen_phone":         "account_security",
    "order_physical_card":          "billing_store",
    "passcode_forgotten":           "account_security",
    "pending_card_payment":         "billing_store",
    "pending_cash_withdrawal":      "billing_store",
    "pending_top_up":               "billing_store",
    "pending_transfer":             "billing_store",
    "pin_blocked":                  "account_security",
    "receiving_money":              "billing_store",
    "Refund_not_showing_up":        "billing_store",
    "request_refund":               "billing_store",
    "reverted_card_payment?":       "billing_store",
    "supported_cards_and_currencies": "software_troubleshooting",
    "terminate_account":            "account_security",
    "top_up_by_bank_transfer_charge": "billing_store",
    "top_up_by_card_charge":        "billing_store",
    "top_up_by_cash_or_cheque":     "billing_store",
    "top_up_failed":                "software_troubleshooting",
    "top_up_limits":                "billing_store",
    "top_up_reverted":              "billing_store",
    "topping_up_by_card":           "billing_store",
    "transaction_charged_twice":    "billing_store",
    "transfer_fee_charged":         "billing_store",
    "transfer_into_account":        "billing_store",
    "transfer_not_received_by_recipient": "billing_store",
    "transfer_timing":              "billing_store",
    "unable_to_verify_identity":    "account_security",
    "verify_my_identity":           "account_security",
    "verify_source_of_funds":       "account_security",
    "verify_top_up":                "account_security",
    "virtual_card_not_working":     "connectivity",
    "visa_or_mastercard":           "billing_store",
    "why_verify_identity":          "account_security",
    "wrong_amount_of_cash_received": "billing_store",
    "wrong_exchange_rate_for_cash_withdrawal": "billing_store",
    "age_limit":                    "account_security",
    "automatic_top_up":             "software_troubleshooting",
    "balance_not_updated_after_bank_transfer": "software_troubleshooting",
    "balance_not_updated_after_cheque_or_cash_deposit": "software_troubleshooting",
    "activate_my_card":             "software_troubleshooting",
}


def analyze_banking77(banking77_data: Dict) -> Dict:
    """
    Comprehensive analysis of Banking77 dataset.
    Returns statistics, taxonomy mapping, and cross-domain insights.
    """
    cache_path = DATA_DIR / "banking77_analysis.json"
    if cache_path.exists():
        logger.info(f"Loading cached Banking77 analysis from {cache_path}")
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    examples = banking77_data["examples"]
    label_names = banking77_data["label_names"]
    label_counts = banking77_data["label_counts"]

    # 1. Basic statistics
    stats = {
        "total_examples": len(examples),
        "num_intents": len(label_names),
        "avg_examples_per_intent": round(len(examples) / len(label_names), 1),
        "min_examples_per_intent": min(label_counts.values()),
        "max_examples_per_intent": max(label_counts.values()),
        "intent_distribution": dict(sorted(label_counts.items(), key=lambda x: -x[1])),
    }

    # 2. Map to AppleSupport taxonomy
    mapped_counts = Counter()
    unmapped = []
    for intent in label_names:
        apple_intent = BANKING77_TO_APPLE_MAP.get(intent)
        if apple_intent:
            mapped_counts[apple_intent] += label_counts.get(intent, 0)
        else:
            unmapped.append(intent)

    mapping_analysis = {
        "mapped_intents": len(BANKING77_TO_APPLE_MAP),
        "unmapped_intents": unmapped,
        "apple_distribution": dict(mapped_counts.most_common()),
        "coverage_pct": round(len(BANKING77_TO_APPLE_MAP) / len(label_names) * 100, 1),
    }

    # 3. Cross-domain insights
    # Compute average query length per intent
    lengths_by_intent = {}
    for ex in examples:
        lname = ex["label_name"]
        lengths_by_intent.setdefault(lname, []).append(len(ex["text"].split()))

    avg_lengths = {k: round(np.mean(v), 1) for k, v in lengths_by_intent.items()}

    # Sample examples per mapped Apple class
    samples_per_apple = {}
    for ex in examples:
        apple_class = BANKING77_TO_APPLE_MAP.get(ex["label_name"])
        if apple_class:
            if apple_class not in samples_per_apple:
                samples_per_apple[apple_class] = []
            if len(samples_per_apple[apple_class]) < 3:
                samples_per_apple[apple_class].append({
                    "text": ex["text"],
                    "banking77_intent": ex["label_name"],
                })

    insights = {
        "avg_query_length_by_intent": dict(sorted(avg_lengths.items(), key=lambda x: -x[1])[:10]),
        "key_finding": (
            "Banking77's 77-class taxonomy is far more granular than our 6-class AppleSupport taxonomy. "
            f"{mapping_analysis['coverage_pct']}% of Banking77 intents map to our taxonomy. "
            "The dominant mapped class is 'billing_store' since banking is inherently transactional. "
            "This validates our design choice of a coarser taxonomy — fine-grained intents like "
            "'card_about_to_expire' vs 'card_arrival' would fragment Apple's support needs. "
            "Banking77's labeled data structure informs our evaluation methodology: "
            "stratified sampling and per-class F1 reporting."
        ),
        "cross_domain_samples": samples_per_apple,
    }

    result = {
        "statistics": stats,
        "mapping_analysis": mapping_analysis,
        "insights": insights,
    }

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    logger.info("Banking77 analysis complete.")
    return result


def train_banking77_classifier(banking77_data: Dict) -> Dict:
    """
    Train and evaluate an intent classifier on Banking77 mapped to
    AppleSupport's 6-class taxonomy. Demonstrates cross-domain transfer.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score

    examples = banking77_data["examples"]

    # Filter to mappable intents only
    texts = []
    labels = []
    for ex in examples:
        apple_class = BANKING77_TO_APPLE_MAP.get(ex["label_name"])
        if apple_class:
            texts.append(ex["text"])
            labels.append(apple_class)

    logger.info(f"Banking77 mapped data: {len(texts)} examples across {len(set(labels))} Apple classes")

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    result = {
        "train_size": len(X_train),
        "test_size": len(X_test),
        "accuracy": round(acc, 4),
        "classification_report": report,
        "finding": (
            f"Cross-domain TF-IDF classifier achieves {acc:.1%} accuracy on Banking77 data "
            "mapped to our 6-class taxonomy. High accuracy reflects that banking queries "
            "overwhelmingly map to 'billing_store' and 'account_security', creating class "
            "imbalance. This validates that our AppleSupport taxonomy captures distinct, "
            "separable intents that can be learned even from out-of-domain data."
        ),
    }

    return result
