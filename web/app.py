"""
web/app.py — FastAPI web application for the AI Support Agent.

Provides:
  - Live interactive agent playground
  - Evaluation dashboard API
  - Golden set explorer API
"""

import json
import logging
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from typing import Optional

# Fix encoding on Windows
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Hiver AI Support Agent", version="1.0")

# ---------------------------------------------------------------------------
# Global state (loaded on startup)
# ---------------------------------------------------------------------------
agent = None
golden_set = None
benchmark = None
banking77_results = None
judge_results = None


class PredictRequest(BaseModel):
    text: str
    max_retrieved: Optional[int] = 3


@app.on_event("startup")
async def startup():
    """Load models and data on startup."""
    global agent, golden_set, benchmark, banking77_results, judge_results

    logger.info("Loading agent components...")

    # Load knowledge base
    kb_path = PROJECT_ROOT / "data" / "knowledge_base.json"
    if not kb_path.exists():
        logger.warning("Knowledge base not found. Run run_pipeline.py first!")
        return

    with open(kb_path, "r", encoding="utf-8") as f:
        kb = json.load(f)

    # Load golden set
    gs_path = PROJECT_ROOT / "data" / "golden_set.json"
    if gs_path.exists():
        with open(gs_path, "r", encoding="utf-8") as f:
            golden_set = json.load(f)

    # Load benchmark results
    bm_path = PROJECT_ROOT / "results" / "benchmark_summary.json"
    if bm_path.exists():
        with open(bm_path, "r", encoding="utf-8") as f:
            benchmark = json.load(f)

    # Load Banking77 results
    b77_path = PROJECT_ROOT / "results" / "banking77_results.json"
    if b77_path.exists():
        with open(b77_path, "r", encoding="utf-8") as f:
            banking77_results = json.load(f)

    # Load judge results
    jr_path = PROJECT_ROOT / "results" / "judge_results.json"
    if jr_path.exists():
        with open(jr_path, "r", encoding="utf-8") as f:
            judge_results = json.load(f)

    # Build agent
    from src.intent_classifier import SemanticClassifier, TFIDFClassifier
    from src.retriever import KnowledgeRetriever
    from src.agent import build_proposed_agent

    semantic_clf = SemanticClassifier()
    try:
        semantic_clf.load()
    except Exception:
        logger.info("No saved semantic model found, training fresh...")
        from src.data_loader import _classify_intent_heuristic
        texts = [item["customer_query"] for item in kb]
        labels = [_classify_intent_heuristic(t)[0] for t in texts]
        if golden_set:
            texts += [g["customer_tweet"] for g in golden_set]
            labels += [g["gold_intent"] for g in golden_set]
        semantic_clf.fit(texts, labels)

    retriever = KnowledgeRetriever(encoder=semantic_clf)
    try:
        retriever.load_index(kb)
    except Exception:
        retriever.index(kb)

    agent = build_proposed_agent(semantic_clf, retriever, use_llm=True)
    logger.info("Agent loaded and ready!")


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.post("/api/predict")
async def predict(request: PredictRequest):
    """Run the full agent pipeline on a customer message."""
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not loaded. Run run_pipeline.py first.")

    result = agent.predict(request.text)
    # Clean numpy types for JSON serialization
    return json.loads(json.dumps(result, default=str))


@app.get("/api/benchmark")
async def get_benchmark():
    """Return benchmark comparison results."""
    if benchmark is None:
        # Try loading individual result files
        results_dir = PROJECT_ROOT / "results"
        agents_data = {}
        for f in results_dir.glob("*_results.json"):
            if f.stem not in ["banking77_results", "judge_results", "benchmark_summary"]:
                with open(f, "r") as fh:
                    agents_data[f.stem] = json.load(fh)
        if agents_data:
            return {"agents": agents_data}
        raise HTTPException(status_code=404, detail="No benchmark results found.")
    return benchmark


@app.get("/api/golden_set")
async def get_golden_set():
    """Return the golden evaluation set."""
    if golden_set is None:
        raise HTTPException(status_code=404, detail="Golden set not found.")
    return {"total": len(golden_set), "examples": golden_set}


@app.get("/api/banking77")
async def get_banking77():
    """Return Banking77 analysis results."""
    if banking77_results is None:
        raise HTTPException(status_code=404, detail="Banking77 results not found.")
    return banking77_results


@app.get("/api/judge")
async def get_judge_results():
    """Return LLM-as-judge evaluation results."""
    if judge_results is None:
        raise HTTPException(status_code=404, detail="Judge results not found.")
    return judge_results


@app.get("/api/quick_tests")
async def get_quick_tests():
    """Return pre-defined test cases for the playground."""
    return [
        {"label": "📱 iOS Update Stuck", "text": "My iPhone is stuck on the Apple logo after updating to iOS 17. It's been like this for 2 hours. Help!"},
        {"label": "🔋 Battery Drain", "text": "Ever since I updated to iOS 16 my iPhone 13 battery drains in 4 hours. This is ridiculous."},
        {"label": "📶 Bluetooth Issue", "text": "My AirPods Pro keep disconnecting from my iPhone every few minutes. I've tried resetting them."},
        {"label": "🔒 Account Locked", "text": "My Apple ID has been disabled for no reason and I can't access any of my data. I need help ASAP!"},
        {"label": "💔 Cracked Screen", "text": "I dropped my iPhone 14 and the screen is completely shattered. What are my repair options?"},
        {"label": "💳 Unauthorized Charge", "text": "I was charged $9.99 for an app I never purchased. I want a refund immediately."},
        {"label": "😡 Angry Customer", "text": "This is the WORST customer service I've ever experienced. I'm switching to Samsung. Your products are garbage!"},
        {"label": "🤔 Ambiguous Query", "text": "my phone is dead"},
        {"label": "🎵 Spotify on Apple", "text": "Can't get Spotify to work on my Apple Watch. It just shows a loading screen."},
        {"label": "☁️ iCloud Storage", "text": "I keep getting notifications that my iCloud storage is full but I've already deleted everything. How do I fix this?"},
    ]


# ---------------------------------------------------------------------------
# Serve static files
# ---------------------------------------------------------------------------

static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main page."""
    index_path = static_dir / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return HTMLResponse("<h1>Hiver AI Support Agent</h1><p>Run run_pipeline.py first.</p>")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
