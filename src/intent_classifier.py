"""
intent_classifier.py — Three-tier intent classification for AppleSupport tweets.

Implements:
  1. TrivialClassifier    — majority-class baseline
  2. TFIDFClassifier      — TF-IDF + Logistic Regression baseline
  3. SemanticClassifier   — sentence-transformers embedding + k-NN (proposed)
"""

import logging
import json
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from src.data_loader import INTENT_TAXONOMY, DATA_DIR

logger = logging.getLogger(__name__)

MODELS_DIR = DATA_DIR.parent / "models"


# =============================================================================
# Base class
# =============================================================================
class IntentClassifier:
    """Abstract base for intent classifiers."""

    def classify(self, text: str) -> Dict:
        """
        Returns: {
            "intent": str,
            "confidence": float,
            "all_scores": dict[str, float]
        }
        """
        raise NotImplementedError

    def classify_batch(self, texts: List[str]) -> List[Dict]:
        return [self.classify(t) for t in texts]


# =============================================================================
# Baseline 1: Trivial — always predicts majority class
# =============================================================================
class TrivialClassifier(IntentClassifier):
    """Always predicts the majority class (software_troubleshooting)."""

    def __init__(self):
        self.majority_class = "software_troubleshooting"

    def classify(self, text: str) -> Dict:
        scores = {k: 0.0 for k in INTENT_TAXONOMY}
        scores[self.majority_class] = 1.0
        return {
            "intent": self.majority_class,
            "confidence": 1.0,
            "all_scores": scores,
        }


# =============================================================================
# Baseline 2: TF-IDF + Logistic Regression
# =============================================================================
class TFIDFClassifier(IntentClassifier):
    """TF-IDF (unigram+bigram) + Logistic Regression classifier."""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words="english",
            min_df=2,
        )
        self.model = LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            C=1.0,
        )
        self.classes: List[str] = []
        self.is_fitted = False

    def fit(self, texts: List[str], labels: List[str]):
        """Train on labelled data."""
        logger.info(f"Training TF-IDF classifier on {len(texts)} examples...")
        X = self.vectorizer.fit_transform(texts)
        self.model.fit(X, labels)
        self.classes = list(self.model.classes_)
        self.is_fitted = True
        logger.info("TF-IDF classifier trained.")

    def classify(self, text: str) -> Dict:
        if not self.is_fitted:
            raise RuntimeError("TFIDFClassifier not fitted. Call fit() first.")
        X = self.vectorizer.transform([text])
        probs = self.model.predict_proba(X)[0]
        scores = {c: float(p) for c, p in zip(self.classes, probs)}
        best = max(scores, key=scores.get)
        return {
            "intent": best,
            "confidence": round(scores[best], 4),
            "all_scores": {k: round(v, 4) for k, v in scores.items()},
        }

    def save(self, path: Optional[Path] = None):
        path = path or MODELS_DIR / "tfidf_classifier.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"vectorizer": self.vectorizer, "model": self.model,
                         "classes": self.classes}, f)
        logger.info(f"TF-IDF classifier saved to {path}")

    def load(self, path: Optional[Path] = None):
        path = path or MODELS_DIR / "tfidf_classifier.pkl"
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.vectorizer = data["vectorizer"]
        self.model = data["model"]
        self.classes = data["classes"]
        self.is_fitted = True
        logger.info(f"TF-IDF classifier loaded from {path}")


# =============================================================================
# Proposed: Semantic Embedding Classifier (all-MiniLM-L6-v2 + k-NN)
# =============================================================================
class SemanticClassifier(IntentClassifier):
    """
    Sentence-transformer embeddings + k-NN classification.
    Uses per-intent prototype embeddings + nearest-neighbor voting.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", k: int = 5):
        logger.info(f"Loading sentence-transformer model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.k = k
        self.embeddings: Optional[np.ndarray] = None
        self.labels: Optional[List[str]] = None
        self.prototype_embeddings: Dict[str, np.ndarray] = {}
        self.is_fitted = False

    def fit(self, texts: List[str], labels: List[str]):
        """Compute embeddings for training data and build prototypes."""
        logger.info(f"Computing embeddings for {len(texts)} training examples...")
        self.embeddings = self.model.encode(texts, show_progress_bar=True,
                                            batch_size=64, normalize_embeddings=True)
        self.labels = labels

        # Compute per-intent prototype (mean embedding)
        intent_vecs = {}
        for emb, label in zip(self.embeddings, labels):
            intent_vecs.setdefault(label, []).append(emb)
        for intent, vecs in intent_vecs.items():
            self.prototype_embeddings[intent] = np.mean(vecs, axis=0)

        self.is_fitted = True
        logger.info(f"Semantic classifier fitted with {len(self.prototype_embeddings)} intent prototypes")

    def classify(self, text: str) -> Dict:
        if not self.is_fitted:
            raise RuntimeError("SemanticClassifier not fitted. Call fit() first.")

        query_emb = self.model.encode([text], normalize_embeddings=True)

        # k-NN voting
        sims = cosine_similarity(query_emb, self.embeddings)[0]
        top_k_idx = np.argsort(sims)[-self.k:]

        votes = {}
        for idx in top_k_idx:
            label = self.labels[idx]
            votes[label] = votes.get(label, 0) + sims[idx]

        # Also compute prototype similarities
        proto_scores = {}
        for intent, proto in self.prototype_embeddings.items():
            proto_scores[intent] = float(cosine_similarity(query_emb, proto.reshape(1, -1))[0][0])

        # Combine: 60% k-NN vote, 40% prototype similarity
        all_intents = set(list(votes.keys()) + list(proto_scores.keys()))
        combined = {}
        max_vote = max(votes.values()) if votes else 1.0
        for intent in all_intents:
            knn_score = votes.get(intent, 0) / max_vote
            proto_score = proto_scores.get(intent, 0)
            combined[intent] = 0.6 * knn_score + 0.4 * proto_score

        # Normalize
        total = sum(combined.values())
        if total > 0:
            combined = {k: v / total for k, v in combined.items()}

        best = max(combined, key=combined.get)
        return {
            "intent": best,
            "confidence": round(combined[best], 4),
            "all_scores": {k: round(v, 4) for k, v in combined.items()},
        }

    def encode_text(self, text: str) -> np.ndarray:
        """Encode a single text. Useful for retriever."""
        return self.model.encode([text], normalize_embeddings=True)[0]

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """Encode multiple texts."""
        return self.model.encode(texts, normalize_embeddings=True,
                                 batch_size=64, show_progress_bar=True)

    def save(self, path: Optional[Path] = None):
        path = path or MODELS_DIR / "semantic_classifier.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "embeddings": self.embeddings,
                "labels": self.labels,
                "prototype_embeddings": self.prototype_embeddings,
                "k": self.k,
            }, f)
        logger.info(f"Semantic classifier data saved to {path}")

    def load(self, path: Optional[Path] = None):
        path = path or MODELS_DIR / "semantic_classifier.pkl"
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.embeddings = data["embeddings"]
        self.labels = data["labels"]
        self.prototype_embeddings = data["prototype_embeddings"]
        self.k = data["k"]
        self.is_fitted = True
        logger.info(f"Semantic classifier data loaded from {path}")


# =============================================================================
# Training helper
# =============================================================================
def train_classifiers(
    golden_set: List[Dict],
    knowledge_base: List[Dict],
) -> Dict[str, IntentClassifier]:
    """
    Train all three classifiers.
    Uses golden set labels + keyword-classified KB examples for training.
    """
    from src.data_loader import _classify_intent_heuristic

    # Build training data from knowledge base (auto-labelled)
    train_texts = []
    train_labels = []
    for item in knowledge_base:
        intent, conf = _classify_intent_heuristic(item["customer_query"])
        if conf >= 0.3:  # minimal confidence filter
            train_texts.append(item["customer_query"])
            train_labels.append(intent)

    # Also add golden set examples
    for item in golden_set:
        train_texts.append(item["customer_tweet"])
        train_labels.append(item["gold_intent"])

    logger.info(f"Training data: {len(train_texts)} examples")

    classifiers = {}

    # 1. Trivial
    classifiers["trivial"] = TrivialClassifier()

    # 2. TF-IDF
    tfidf = TFIDFClassifier()
    tfidf.fit(train_texts, train_labels)
    tfidf.save()
    classifiers["tfidf"] = tfidf

    # 3. Semantic
    semantic = SemanticClassifier()
    semantic.fit(train_texts, train_labels)
    semantic.save()
    classifiers["semantic"] = semantic

    return classifiers
