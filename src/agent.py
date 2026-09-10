"""
agent.py — Main AI Support Agent pipeline orchestrator.

Combines intent classification, semantic retrieval, escalation decision,
and grounded response generation into a single predict() call.
"""

import logging
from typing import Dict, List, Optional

from src.intent_classifier import IntentClassifier, SemanticClassifier
from src.retriever import KnowledgeRetriever
from src.escalation_engine import (
    EscalationEngine,
    TrivialEscalationEngine,
    KeywordEscalationEngine,
)
from src.response_generator import (
    ResponseGenerator,
    TrivialResponseGenerator,
    NearestNeighborResponseGenerator,
)

logger = logging.getLogger(__name__)


class SupportAgent:
    """
    Complete AI Support Agent pipeline.

    Given a customer tweet, produces:
      - Predicted intent + confidence
      - Escalation decision + stated reason
      - Retrieved historical cases
      - Grounded draft reply
    """

    def __init__(
        self,
        classifier: IntentClassifier,
        retriever: KnowledgeRetriever,
        escalation_engine: EscalationEngine,
        response_generator,
        agent_name: str = "proposed",
    ):
        self.classifier = classifier
        self.retriever = retriever
        self.escalation_engine = escalation_engine
        self.response_generator = response_generator
        self.agent_name = agent_name

    def predict(self, customer_text: str) -> Dict:
        """
        Full pipeline prediction for a single customer message.

        Returns:
            {
                "customer_text": str,
                "intent": str,
                "intent_confidence": float,
                "intent_all_scores": dict,
                "escalation_decision": str,
                "escalation_reason": str,
                "escalation_factors": list,
                "risk_score": float,
                "retrieved_cases": list,
                "response": str,
                "response_method": str,
                "grounding_sources": list,
                "agent_name": str,
            }
        """
        # 1. Classify intent
        intent_result = self.classifier.classify(customer_text)

        # 2. Retrieve similar historical cases
        retrieved = self.retriever.retrieve(customer_text, top_k=3)
        best_similarity = retrieved[0]["similarity"] if retrieved else 0.0

        # 3. Escalation decision
        esc_result = self.escalation_engine.decide(
            text=customer_text,
            intent=intent_result["intent"],
            confidence=intent_result["confidence"],
            retrieval_similarity=best_similarity,
        )

        # 4. Generate grounded response
        gen_result = self.response_generator.generate(
            customer_text=customer_text,
            intent=intent_result["intent"],
            escalation_decision=esc_result["decision"],
            escalation_reason=esc_result["reason"],
            retrieved=retrieved,
        )

        return {
            "customer_text": customer_text,
            "intent": intent_result["intent"],
            "intent_confidence": intent_result["confidence"],
            "intent_all_scores": intent_result["all_scores"],
            "escalation_decision": esc_result["decision"],
            "escalation_reason": esc_result["reason"],
            "escalation_factors": esc_result["factors"],
            "risk_score": esc_result["risk_score"],
            "retrieved_cases": retrieved,
            "response": gen_result["response"],
            "response_method": gen_result["method"],
            "grounding_sources": gen_result["grounding_sources"],
            "agent_name": self.agent_name,
        }

    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """Predict for a batch of customer messages."""
        return [self.predict(text) for text in texts]


# ---------------------------------------------------------------------------
# Agent factory functions
# ---------------------------------------------------------------------------

def build_trivial_agent(retriever: KnowledgeRetriever) -> SupportAgent:
    """Build Baseline 1: trivial majority-class agent."""
    from src.intent_classifier import TrivialClassifier
    return SupportAgent(
        classifier=TrivialClassifier(),
        retriever=retriever,
        escalation_engine=TrivialEscalationEngine(),
        response_generator=TrivialResponseGenerator(),
        agent_name="trivial_baseline",
    )


def build_tfidf_agent(
    tfidf_classifier,
    retriever: KnowledgeRetriever,
) -> SupportAgent:
    """Build Baseline 2: TF-IDF + keyword escalation + 1-NN response."""
    return SupportAgent(
        classifier=tfidf_classifier,
        retriever=retriever,
        escalation_engine=KeywordEscalationEngine(),
        response_generator=NearestNeighborResponseGenerator(),
        agent_name="tfidf_baseline",
    )


def build_proposed_agent(
    semantic_classifier: SemanticClassifier,
    retriever: KnowledgeRetriever,
    use_llm: bool = True,
) -> SupportAgent:
    """Build the proposed AI agent with semantic classification + RAG + LLM."""
    return SupportAgent(
        classifier=semantic_classifier,
        retriever=retriever,
        escalation_engine=EscalationEngine(confidence_threshold=0.35),
        response_generator=ResponseGenerator(use_llm=use_llm),
        agent_name="proposed_agent",
    )
