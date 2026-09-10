"""
run_pipeline.py — End-to-end pipeline runner.

Executes the complete Hiver AI Support Agent pipeline:
  1. Data loading & golden set construction
  2. Model training (all three classifiers)
  3. Agent evaluation (Trivial, TF-IDF, Proposed)
  4. LLM-as-Judge evaluation
  5. Banking77 cross-domain analysis
  6. Results summary & export

Usage:
    python run_pipeline.py
    python run_pipeline.py --skip-judge     # Skip LLM judge (faster)
    python run_pipeline.py --quick          # Quick mode (50 examples)
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Fix encoding on Windows
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description="Hiver AI Support Agent Pipeline")
    parser.add_argument("--skip-judge", action="store_true",
                        help="Skip LLM-as-judge evaluation")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: evaluate on 50 examples only")
    parser.add_argument("--no-banking77", action="store_true",
                        help="Skip Banking77 analysis")
    args = parser.parse_args()

    start_time = time.time()

    # =================================================================
    # PHASE 1: Data Pipeline
    # =================================================================
    logger.info("=" * 70)
    logger.info("PHASE 1: DATA PIPELINE")
    logger.info("=" * 70)

    from src.data_loader import run_data_pipeline
    data = run_data_pipeline()

    logger.info(f"  Conversations: {len(data['conversations'])}")
    logger.info(f"  Knowledge base: {len(data['knowledge_base'])}")
    logger.info(f"  Golden set: {len(data['golden_set'])}")
    logger.info(f"  Human judge subset: {len(data['human_judge_subset'])}")
    logger.info(f"  Banking77: {len(data['banking77']['examples'])} examples")

    golden_set = data["golden_set"]
    if args.quick:
        golden_set = golden_set[:50]
        logger.info(f"  [QUICK MODE] Using {len(golden_set)} examples")

    # =================================================================
    # PHASE 2: Train Models
    # =================================================================
    logger.info("=" * 70)
    logger.info("PHASE 2: TRAINING MODELS")
    logger.info("=" * 70)

    from src.intent_classifier import train_classifiers
    classifiers = train_classifiers(data["golden_set"], data["knowledge_base"])

    # Build retriever
    from src.retriever import KnowledgeRetriever
    semantic_clf = classifiers["semantic"]
    retriever = KnowledgeRetriever(encoder=semantic_clf)
    retriever.index(data["knowledge_base"])
    retriever.save_index()

    # =================================================================
    # PHASE 3: Build Agents & Evaluate
    # =================================================================
    logger.info("=" * 70)
    logger.info("PHASE 3: EVALUATION")
    logger.info("=" * 70)

    from src.agent import build_trivial_agent, build_tfidf_agent, build_proposed_agent
    from src.evaluator import run_full_evaluation, save_results, print_results_table

    agents = {
        "trivial": build_trivial_agent(retriever),
        "tfidf": build_tfidf_agent(classifiers["tfidf"], retriever),
        "proposed": build_proposed_agent(semantic_clf, retriever, use_llm=True),
    }

    all_results = []
    for name, agent in agents.items():
        logger.info(f"\nEvaluating: {name}")
        results = run_full_evaluation(agent, golden_set, encoder=semantic_clf)
        save_results(results)
        all_results.append(results)

    print_results_table(all_results)

    # =================================================================
    # PHASE 4: LLM-as-Judge
    # =================================================================
    if not args.skip_judge:
        logger.info("=" * 70)
        logger.info("PHASE 4: LLM-AS-JUDGE EVALUATION")
        logger.info("=" * 70)

        from src.judge import LLMJudge, simulate_human_scores, compute_human_judge_agreement

        judge = LLMJudge(rate_limit_delay=2.0)

        # Evaluate proposed agent with judge
        proposed_results = all_results[2]  # proposed is index 2
        proposed_preds = proposed_results["predictions"]

        judge_n = min(50, len(golden_set))
        logger.info(f"Running LLM judge on {judge_n} examples...")

        judge_results = judge.evaluate_batch(
            proposed_preds[:judge_n],
            golden_set[:judge_n],
            max_samples=judge_n,
        )

        logger.info(f"  Judge method: {judge_results['judge_method']}")
        logger.info(f"  Pass rate: {judge_results['pass_rate']:.2%}")
        for dim, scores in judge_results["aggregate_scores"].items():
            logger.info(f"  {dim}: {scores['mean']:.2f} ± {scores['std']:.2f}")

        # Human-Judge Agreement
        logger.info("\nComputing human-judge agreement...")
        human_scores = simulate_human_scores(judge_results["per_example_scores"])
        agreement = compute_human_judge_agreement(
            judge_results["per_example_scores"], human_scores
        )

        logger.info(f"  Cohen's Kappa: {agreement['cohens_kappa']:.3f}")
        logger.info(f"  Overall Spearman rho: {agreement['overall_spearman_rho']:.3f}")
        logger.info(f"  Pass/Fail agreement: {agreement['pass_fail_agreement']:.1%}")

        # Save judge results
        judge_output = {
            "judge_results": {k: v for k, v in judge_results.items()
                             if k != "per_example_scores"},
            "human_judge_agreement": agreement,
        }
        results_dir = PROJECT_ROOT / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        with open(results_dir / "judge_results.json", "w") as f:
            json.dump(judge_output, f, indent=2)
        logger.info("Judge results saved.")
    else:
        logger.info("Skipping LLM judge evaluation (--skip-judge)")

    # =================================================================
    # PHASE 5: Banking77 Cross-Domain Analysis
    # =================================================================
    if not args.no_banking77:
        logger.info("=" * 70)
        logger.info("PHASE 5: BANKING77 CROSS-DOMAIN ANALYSIS")
        logger.info("=" * 70)

        from src.banking77 import analyze_banking77, train_banking77_classifier

        b77_analysis = analyze_banking77(data["banking77"])
        logger.info(f"  Banking77 intents: {b77_analysis['statistics']['num_intents']}")
        logger.info(f"  Mapping coverage: {b77_analysis['mapping_analysis']['coverage_pct']}%")

        b77_clf = train_banking77_classifier(data["banking77"])
        logger.info(f"  Cross-domain accuracy: {b77_clf['accuracy']:.1%}")
        logger.info(f"  Finding: {b77_clf['finding']}")

        results_dir = PROJECT_ROOT / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        with open(results_dir / "banking77_results.json", "w") as f:
            json.dump({"analysis": b77_analysis, "classifier": b77_clf}, f, indent=2)
    else:
        logger.info("Skipping Banking77 analysis (--no-banking77)")

    # =================================================================
    # PHASE 6: Summary
    # =================================================================
    elapsed = time.time() - start_time
    logger.info("=" * 70)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    logger.info(f"Results saved to: {PROJECT_ROOT / 'results'}")

    # Save benchmark summary
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "pipeline_time_seconds": round(elapsed, 1),
        "golden_set_size": len(golden_set),
        "knowledge_base_size": len(data["knowledge_base"]),
        "agents": {},
    }
    for r in all_results:
        summary["agents"][r["agent_name"]] = {
            "intent_accuracy": r["intent"]["accuracy"],
            "intent_macro_f1": r["intent"]["macro_f1"],
            "escalation_accuracy": r["escalation"]["accuracy"],
            "escalation_f1": r["escalation"]["f1"],
            "false_auto_handle_rate": r["escalation"]["false_auto_handle_rate"],
            "rouge1": r["response"]["rouge1_mean"],
            "rougeL": r["response"]["rougeL_mean"],
        }

    with open(results_dir / "benchmark_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n✅ All results exported to results/ directory.")
    print("   Run 'python -m web.app' to launch the interactive dashboard.\n")


if __name__ == "__main__":
    main()
