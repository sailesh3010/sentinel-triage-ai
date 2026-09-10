"""
retriever.py — Semantic retrieval (RAG) over the AppleSupport knowledge base.

Uses all-MiniLM-L6-v2 embeddings + cosine similarity to find the top-k
historically similar customer queries and their brand-verified resolutions.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.data_loader import DATA_DIR

logger = logging.getLogger(__name__)


class KnowledgeRetriever:
    """
    Semantic retrieval over the AppleSupport knowledge base.
    Indexes customer queries with sentence-transformer embeddings and
    retrieves top-k similar historical resolution pairs.
    """

    def __init__(self, encoder=None):
        """
        Args:
            encoder: A SemanticClassifier instance (reuses its model to avoid
                     loading the transformer twice).
        """
        self.encoder = encoder
        self.kb: List[Dict] = []
        self.embeddings: Optional[np.ndarray] = None
        self.is_indexed = False

    def index(self, knowledge_base: List[Dict]):
        """Build the vector index over the knowledge base."""
        self.kb = knowledge_base
        texts = [item["customer_query"] for item in knowledge_base]

        logger.info(f"Indexing {len(texts)} knowledge base entries...")
        if self.encoder is not None:
            self.embeddings = self.encoder.encode_batch(texts)
        else:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("all-MiniLM-L6-v2")
            self.embeddings = model.encode(texts, normalize_embeddings=True,
                                            batch_size=64, show_progress_bar=True)

        self.is_indexed = True
        logger.info("Knowledge base indexed.")

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Retrieve top-k similar historical cases for a given customer query.

        Returns list of dicts:
          - customer_query: the historical customer message
          - support_response: the brand's actual reply
          - similarity: cosine similarity score
          - id: conversation id
        """
        if not self.is_indexed:
            raise RuntimeError("Retriever not indexed. Call index() first.")

        if self.encoder is not None:
            query_emb = self.encoder.encode_text(query).reshape(1, -1)
        else:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("all-MiniLM-L6-v2")
            query_emb = model.encode([query], normalize_embeddings=True)

        sims = cosine_similarity(query_emb, self.embeddings)[0]
        top_k_idx = np.argsort(sims)[-top_k:][::-1]

        results = []
        for idx in top_k_idx:
            results.append({
                "customer_query": self.kb[idx]["customer_query"],
                "support_response": self.kb[idx]["support_response"],
                "similarity": round(float(sims[idx]), 4),
                "id": self.kb[idx]["id"],
            })

        return results

    def save_index(self, path: Optional[Path] = None):
        """Save embeddings to disk."""
        path = path or DATA_DIR.parent / "models" / "kb_embeddings.npy"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, self.embeddings)
        logger.info(f"KB embeddings saved to {path}")

    def load_index(self, knowledge_base: List[Dict], path: Optional[Path] = None):
        """Load pre-computed embeddings."""
        path = path or DATA_DIR.parent / "models" / "kb_embeddings.npy"
        self.kb = knowledge_base
        self.embeddings = np.load(path)
        self.is_indexed = True
        logger.info(f"KB embeddings loaded from {path} ({self.embeddings.shape})")
