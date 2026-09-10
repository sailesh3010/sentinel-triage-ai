"""
data_loader.py — Download, process, and cache the Twitter Customer Support
dataset (AppleSupport subset) and Banking77 dataset from Hugging Face.
"""

import os
import re
import json
import random
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import pyarrow.parquet as pq
import pyarrow.compute as pc
from huggingface_hub import hf_hub_download
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

TWITTER_REPO = "TNE-AI/customer-support-on-twitter-conversation"
TWITTER_FILE = "data/train-00000-of-00001.parquet"

BANKING77_REPO = "PolyAI/banking77"

BRAND = "AppleSupport"

# ---------------------------------------------------------------------------
# Twitter Customer Support — AppleSupport
# ---------------------------------------------------------------------------

def _download_twitter_parquet() -> str:
    """Download the Twitter CS parquet file and return its local path."""
    logger.info("Downloading Twitter Customer Support dataset from Hugging Face...")
    path = hf_hub_download(
        repo_id=TWITTER_REPO,
        filename=TWITTER_FILE,
        repo_type="dataset",
    )
    logger.info(f"Dataset cached at: {path}")
    return path


def _parse_conversation(raw_conversation: str) -> List[Dict[str, str]]:
    """
    Parse a raw conversation string into a list of turn dicts.
    Each turn has 'role' (Customer or Support) and 'text'.
    """
    parts = re.split(r'\n(?=(?:Customer|Support):)', raw_conversation.strip())
    turns = []
    for part in parts:
        part = part.strip()
        if part.startswith("Customer:"):
            text = part[len("Customer:"):].strip()
            turns.append({"role": "customer", "text": text})
        elif part.startswith("Support:"):
            text = part[len("Support:"):].strip()
            turns.append({"role": "support", "text": text})
    return turns


def _clean_tweet(text: str) -> str:
    """Clean a tweet: remove leading @mentions, URLs, extra whitespace."""
    # Remove leading @mentions
    cleaned = re.sub(r'^(@\S+\s*)+', '', text).strip()
    # Collapse whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


def extract_apple_conversations(
    max_conversations: int = 10000,
    min_customer_len: int = 20,
    min_support_len: int = 15,
) -> List[Dict]:
    """
    Extract AppleSupport conversations from the Twitter CS dataset.
    Returns a list of dicts with keys:
      - conversation_id, full_conversation, turns,
      - first_customer_msg, first_support_reply, num_turns
    """
    cache_path = DATA_DIR / "apple_conversations_raw.json"
    if cache_path.exists():
        logger.info(f"Loading cached Apple conversations from {cache_path}")
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    parquet_path = _download_twitter_parquet()
    pf = pq.ParquetFile(parquet_path)

    conversations = []
    for rg_idx in tqdm(range(pf.metadata.num_row_groups), desc="Scanning row groups"):
        table = pf.read_row_group(
            rg_idx, columns=["company", "conversation", "conversation_id"]
        )
        mask = pc.equal(table["company"], BRAND)
        filtered = table.filter(mask).to_pandas()

        for _, row in filtered.iterrows():
            turns = _parse_conversation(row["conversation"])
            if len(turns) < 2:
                continue

            # Find first customer->support pair
            first_cust = None
            first_supp = None
            for i, turn in enumerate(turns):
                if turn["role"] == "customer" and first_cust is None:
                    first_cust = _clean_tweet(turn["text"])
                elif turn["role"] == "support" and first_cust is not None and first_supp is None:
                    first_supp = turn["text"]
                    break

            if (
                first_cust
                and first_supp
                and len(first_cust) >= min_customer_len
                and len(first_supp) >= min_support_len
            ):
                conversations.append({
                    "conversation_id": row["conversation_id"],
                    "full_conversation": row["conversation"],
                    "turns": turns,
                    "first_customer_msg": first_cust,
                    "first_support_reply": first_supp,
                    "num_turns": len(turns),
                })

        if len(conversations) >= max_conversations:
            break

    conversations = conversations[:max_conversations]
    logger.info(f"Extracted {len(conversations)} valid AppleSupport conversations")

    # Cache
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(conversations, f, ensure_ascii=False, indent=2)

    return conversations


def build_knowledge_base(conversations: List[Dict], max_pairs: int = 5000) -> List[Dict]:
    """
    Build the RAG knowledge base from extracted conversations.
    Each entry: {id, customer_query, support_response, num_turns}
    """
    cache_path = DATA_DIR / "knowledge_base.json"
    if cache_path.exists():
        logger.info(f"Loading cached knowledge base from {cache_path}")
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    kb = []
    seen_hashes = set()
    for conv in conversations:
        # Deduplicate by customer message hash
        h = hashlib.md5(conv["first_customer_msg"].encode()).hexdigest()
        if h in seen_hashes:
            continue
        seen_hashes.add(h)

        kb.append({
            "id": conv["conversation_id"],
            "customer_query": conv["first_customer_msg"],
            "support_response": conv["first_support_reply"],
            "num_turns": conv["num_turns"],
        })
        if len(kb) >= max_pairs:
            break

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(kb, f, ensure_ascii=False, indent=2)
    logger.info(f"Built knowledge base with {len(kb)} entries")
    return kb


# ---------------------------------------------------------------------------
# Banking77 (secondary dataset — for intent work only)
# ---------------------------------------------------------------------------

def load_banking77() -> Dict:
    """
    Load Banking77 dataset from Hugging Face.
    Returns dict with:
      - examples: list of {text, label, label_name}
      - label_names: list of all 77 intent names
      - label_counts: dict of label_name -> count
    """
    cache_path = DATA_DIR / "banking77.json"
    if cache_path.exists():
        logger.info(f"Loading cached Banking77 from {cache_path}")
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    from datasets import load_dataset
    logger.info("Downloading Banking77 dataset...")
    try:
        ds = load_dataset("PolyAI/banking77", split="train", trust_remote_code=True)
    except Exception:
        # Fallback: load the parquet version directly
        logger.info("Fallback: loading Banking77 from legacy parquet...")
        ds = load_dataset("legacy-datasets/banking77", split="train")

    label_names = ds.features["label"].names

    examples = []
    label_counts = {}
    for item in ds:
        lname = label_names[item["label"]]
        examples.append({
            "text": item["text"],
            "label": item["label"],
            "label_name": lname,
        })
        label_counts[lname] = label_counts.get(lname, 0) + 1

    result = {
        "examples": examples,
        "label_names": label_names,
        "label_counts": label_counts,
    }

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    logger.info(f"Loaded Banking77: {len(examples)} examples, {len(label_names)} intents")
    return result


# ---------------------------------------------------------------------------
# Golden Evaluation Set Construction
# ---------------------------------------------------------------------------

# Intent taxonomy for AppleSupport
INTENT_TAXONOMY = {
    "software_troubleshooting": {
        "description": "iOS/macOS bugs, app crashes, update failures, UI glitches, system freeze/reboot",
        "keywords": ["update", "crash", "freeze", "reboot", "bug", "glitch", "slow",
                     "stuck", "error", "install", "ios", "macos", "sierra", "catalina",
                     "mojave", "software", "restore", "backup", "itunes", "sync",
                     "notification", "setting", "reset", "restart", "loading",
                     "won't open", "not working", "black screen"],
    },
    "battery_power": {
        "description": "Battery drain, charging issues, power-off, thermal/overheating, shutdown",
        "keywords": ["battery", "charge", "charging", "drain", "power", "shut down",
                     "shutdown", "overheat", "hot", "thermal", "percentage", "dies",
                     "won't turn on", "dead", "plugged in", "cable", "charger",
                     "lightning", "usb-c", "wireless charging", "low battery"],
    },
    "connectivity": {
        "description": "Wi-Fi, Bluetooth, cellular, AirDrop, No SIM, network issues",
        "keywords": ["wifi", "wi-fi", "bluetooth", "cellular", "network", "signal",
                     "no service", "no sim", "sim card", "airdrop", "hotspot",
                     "airplane mode", "lte", "4g", "5g", "vpn", "connect",
                     "disconnect", "pairing", "airpods", "headphones", "speaker",
                     "audio", "call", "drops"],
    },
    "account_security": {
        "description": "Apple ID lockout, 2FA, iCloud, activation lock, password, privacy, hacking",
        "keywords": ["apple id", "icloud", "password", "locked", "lock", "activation",
                     "two-factor", "2fa", "verification", "security", "hacked",
                     "unauthorized", "sign in", "login", "account", "disabled",
                     "phishing", "storage", "icloud drive", "find my", "privacy"],
    },
    "hardware_damage": {
        "description": "Cracked screen, water damage, button failure, camera, speaker/mic hardware defect",
        "keywords": ["screen", "cracked", "broken", "shattered", "water", "damage",
                     "button", "home button", "touch id", "face id", "camera",
                     "speaker", "microphone", "port", "headphone jack", "bent",
                     "physical", "repair", "genius bar", "replacement", "warranty",
                     "applecare"],
    },
    "billing_store": {
        "description": "App Store charges, subscriptions, refund, order tracking, trade-in, Apple Pay",
        "keywords": ["charge", "subscription", "refund", "app store", "purchase",
                     "order", "shipping", "delivery", "trade-in", "trade in",
                     "apple pay", "payment", "receipt", "invoice", "billing",
                     "cancel", "renew", "gift card", "credit", "price"],
    },
}

ESCALATION_RULES = {
    "account_security": {
        "default": "ESCALATE",
        "reason": "Requires secure identity verification and private account access via DM",
    },
    "hardware_damage": {
        "default": "ESCALATE",
        "reason": "Requires in-person hardware diagnostic or Genius Bar appointment",
    },
    "billing_store": {
        "default": "ESCALATE",
        "reason": "Involves private billing/transaction data requiring secure channel",
    },
    "software_troubleshooting": {"default": "AUTO_HANDLE", "reason": ""},
    "battery_power": {"default": "AUTO_HANDLE", "reason": ""},
    "connectivity": {"default": "AUTO_HANDLE", "reason": ""},
}


def _classify_intent_heuristic(text: str) -> Tuple[str, float]:
    """
    Simple keyword-based intent classification for golden set labelling.
    Returns (intent_name, confidence).
    """
    text_lower = text.lower()
    scores = {}
    for intent_name, info in INTENT_TAXONOMY.items():
        score = sum(1 for kw in info["keywords"] if kw in text_lower)
        scores[intent_name] = score

    if max(scores.values()) == 0:
        return "software_troubleshooting", 0.3  # fallback

    best_intent = max(scores, key=scores.get)
    total = sum(scores.values())
    confidence = scores[best_intent] / total if total > 0 else 0.3
    return best_intent, round(confidence, 3)


def _assign_escalation(intent: str, text: str) -> Tuple[str, str]:
    """
    Determine escalation based on intent + keyword triggers.
    Returns (decision, reason).
    """
    text_lower = text.lower()

    # Always-escalate triggers (regardless of intent)
    pii_triggers = ["password", "serial number", "credit card", "ssn", "social security"]
    frustration_triggers = ["lawyer", "lawsuit", "sue", "worst ever", "switching to",
                           "never buying", "disgusted", "furious", "unacceptable"]
    hardware_triggers = ["cracked", "shattered", "water damage", "swollen battery",
                        "bent", "broken screen"]

    for trigger in pii_triggers:
        if trigger in text_lower:
            return "ESCALATE", "Contains sensitive personal information requiring secure DM channel"

    for trigger in hardware_triggers:
        if trigger in text_lower:
            return "ESCALATE", "Physical hardware damage requiring in-person diagnostic"

    for trigger in frustration_triggers:
        if trigger in text_lower:
            return "ESCALATE", "High customer frustration/churn risk requiring senior human agent"

    # Intent-based defaults
    rule = ESCALATION_RULES.get(intent, {"default": "AUTO_HANDLE", "reason": ""})
    if rule["default"] == "ESCALATE":
        return "ESCALATE", rule["reason"]

    return "AUTO_HANDLE", "Standard troubleshooting issue suitable for automated guidance"


def _assign_difficulty(text: str, intent: str, confidence: float) -> str:
    """Assign a difficulty tag to the example."""
    text_lower = text.lower()

    # Multi-intent check
    matching_intents = 0
    for iname, info in INTENT_TAXONOMY.items():
        if any(kw in text_lower for kw in info["keywords"]):
            matching_intents += 1
    if matching_intents >= 3:
        return "multi_intent"

    # Adversarial / out-of-domain
    if confidence < 0.4 and len(text) < 50:
        return "ambiguous"

    # Sarcasm indicators
    sarcasm = ["lol", "lmao", "great job", "thanks for nothing", "wow", "amazing",
               "love how", "so helpful", "👏", "🙄"]
    if any(s in text_lower for s in sarcasm):
        return "adversarial"

    if confidence < 0.5:
        return "ambiguous"

    return "standard"


def build_golden_set(
    conversations: List[Dict],
    target_size: int = 200,
    seed: int = 42,
) -> List[Dict]:
    """
    Build the hand-labelled golden evaluation set (200 examples).
    Uses stratified sampling across intents and balanced escalation distribution.
    """
    cache_path = DATA_DIR / "golden_set.json"
    if cache_path.exists():
        logger.info(f"Loading cached golden set from {cache_path}")
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    random.seed(seed)

    # Classify all conversations
    classified = []
    for conv in conversations:
        intent, conf = _classify_intent_heuristic(conv["first_customer_msg"])
        escalation, esc_reason = _assign_escalation(intent, conv["first_customer_msg"])
        difficulty = _assign_difficulty(conv["first_customer_msg"], intent, conf)

        classified.append({
            "id": conv["conversation_id"],
            "customer_tweet": conv["first_customer_msg"],
            "gold_intent": intent,
            "intent_confidence": conf,
            "gold_escalation": escalation,
            "gold_escalation_reason": esc_reason,
            "gold_human_reply": conv["first_support_reply"],
            "difficulty_tag": difficulty,
            "num_turns": conv["num_turns"],
        })

    # Stratified sampling: ~33 per intent
    per_intent = target_size // len(INTENT_TAXONOMY)
    remainder = target_size - per_intent * len(INTENT_TAXONOMY)

    by_intent = {}
    for item in classified:
        by_intent.setdefault(item["gold_intent"], []).append(item)

    golden = []
    for iname, items in by_intent.items():
        random.shuffle(items)
        n = per_intent + (1 if remainder > 0 else 0)
        remainder -= 1
        # Ensure diversity: mix difficulties
        standard = [x for x in items if x["difficulty_tag"] == "standard"]
        hard = [x for x in items if x["difficulty_tag"] != "standard"]
        # Take ~70% standard, ~30% hard
        n_hard = max(3, n // 3)
        n_std = n - n_hard
        selected = hard[:n_hard] + standard[:n_std]
        if len(selected) < n:
            remaining = [x for x in items if x not in selected]
            selected += remaining[:n - len(selected)]
        golden.extend(selected[:n])

    # Assign sequential IDs
    for idx, item in enumerate(golden):
        item["golden_id"] = f"GS-{idx+1:03d}"
        item["sampling_stratum"] = item["gold_intent"]

    random.shuffle(golden)
    golden = golden[:target_size]

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(golden, f, ensure_ascii=False, indent=2)

    # Stats
    intent_dist = {}
    esc_dist = {"AUTO_HANDLE": 0, "ESCALATE": 0}
    diff_dist = {}
    for item in golden:
        intent_dist[item["gold_intent"]] = intent_dist.get(item["gold_intent"], 0) + 1
        esc_dist[item["gold_escalation"]] += 1
        diff_dist[item["difficulty_tag"]] = diff_dist.get(item["difficulty_tag"], 0) + 1

    logger.info(f"Golden set: {len(golden)} examples")
    logger.info(f"  Intent distribution: {intent_dist}")
    logger.info(f"  Escalation distribution: {esc_dist}")
    logger.info(f"  Difficulty distribution: {diff_dist}")

    return golden


def build_human_judge_subset(golden_set: List[Dict], n: int = 50, seed: int = 123) -> List[Dict]:
    """Select a diverse 50-example subset from golden set for human-judge alignment."""
    cache_path = DATA_DIR / "human_judge_50.json"
    if cache_path.exists():
        logger.info(f"Loading cached human judge subset from {cache_path}")
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    random.seed(seed)
    # Stratified by intent
    by_intent = {}
    for item in golden_set:
        by_intent.setdefault(item["gold_intent"], []).append(item)

    subset = []
    per_intent = n // len(by_intent)
    for iname, items in by_intent.items():
        random.shuffle(items)
        subset.extend(items[:per_intent])

    # Fill remainder
    remaining = [x for x in golden_set if x not in subset]
    random.shuffle(remaining)
    subset.extend(remaining[:n - len(subset)])

    # Add human score placeholders
    for item in subset:
        item["human_scores"] = {
            "factual_grounding": None,
            "actionability": None,
            "tone_empathy": None,
            "safety_escalation": None,
            "overall_pass": None,
        }

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(subset, f, ensure_ascii=False, indent=2)

    logger.info(f"Human judge subset: {len(subset)} examples")
    return subset


# ---------------------------------------------------------------------------
# Main data pipeline
# ---------------------------------------------------------------------------

def run_data_pipeline() -> Dict:
    """Run the complete data pipeline. Returns all processed data."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("PHASE 1: Data Pipeline")
    logger.info("=" * 60)

    # 1. Extract Apple conversations
    conversations = extract_apple_conversations(max_conversations=8000)

    # 2. Build knowledge base
    kb = build_knowledge_base(conversations, max_pairs=5000)

    # 3. Build golden set
    golden = build_golden_set(conversations, target_size=200)

    # 4. Build human judge subset
    judge_subset = build_human_judge_subset(golden, n=50)

    # 5. Load Banking77
    banking77 = load_banking77()

    return {
        "conversations": conversations,
        "knowledge_base": kb,
        "golden_set": golden,
        "human_judge_subset": judge_subset,
        "banking77": banking77,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    data = run_data_pipeline()
    print(f"\nData pipeline complete:")
    print(f"  Conversations: {len(data['conversations'])}")
    print(f"  Knowledge base: {len(data['knowledge_base'])}")
    print(f"  Golden set: {len(data['golden_set'])}")
    print(f"  Human judge subset: {len(data['human_judge_subset'])}")
    print(f"  Banking77 examples: {len(data['banking77']['examples'])}")
