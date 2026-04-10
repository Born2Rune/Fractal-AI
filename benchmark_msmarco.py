"""
MS MARCO Benchmark for Fractal AI Memory Layer

Evaluates Fractal AI on MS MARCO passage ranking task.
Industry standard for passage retrieval evaluation.
"""

import sys
import time
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

from enhancements.memory_layer import MemoryLayer

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("datasets not installed. Run: pip install datasets")

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress
    console = Console()
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


def calculate_mrr(qrels: Dict, results: Dict, k: int = 10) -> float:
    """
    Calculate Mean Reciprocal Rank @ k
    
    Args:
        qrels: Ground truth {query_id: {doc_id: relevance}}
        results: Predictions {query_id: {doc_id: score}}
        k: Cutoff rank
        
    Returns:
        MRR@k score
    """
    reciprocal_ranks = []
    
    for query_id, doc_scores in results.items():
        if query_id not in qrels:
            continue
        
        # Get relevant docs for this query
        relevant_docs = set(doc_id for doc_id, rel in qrels[query_id].items() if rel > 0)
        
        if not relevant_docs:
            continue
        
        # Sort predictions by score
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Find rank of first relevant document
        for rank, (doc_id, score) in enumerate(sorted_docs[:k], 1):
            if doc_id in relevant_docs:
                reciprocal_ranks.append(1.0 / rank)
                break
        else:
            reciprocal_ranks.append(0.0)
    
    return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0


def calculate_recall(qrels: Dict, results: Dict, k: int = 1000) -> float:
    """
    Calculate Recall @ k
    
    Args:
        qrels: Ground truth {query_id: {doc_id: relevance}}
        results: Predictions {query_id: {doc_id: score}}
        k: Cutoff rank
        
    Returns:
        Recall@k score
    """
    recalls = []
    
    for query_id, doc_scores in results.items():
        if query_id not in qrels:
            continue
        
        # Get relevant docs for this query
        relevant_docs = set(doc_id for doc_id, rel in qrels[query_id].items() if rel > 0)
        
        if not relevant_docs:
            continue
        
        # Get top-k predictions
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        top_k_docs = set(doc_id for doc_id, score in sorted_docs[:k])
        
        # Calculate recall
        retrieved_relevant = len(relevant_docs & top_k_docs)
        recall = retrieved_relevant / len(relevant_docs)
        recalls.append(recall)
    
    return np.mean(recalls) if recalls else 0.0


def evaluate_msmarco_dev(
    num_queries: int = 100,
    device: str = "cuda"
) -> Dict[str, float]:
    """
    Evaluate Fractal AI on MS MARCO dev set.
    
    Args:
        num_queries: Number of queries to evaluate (full dev has 6980)
        device: Device to run on
        
    Returns:
        Evaluation metrics dict
    """
    if not DATASETS_AVAILABLE:
        print("ERROR: datasets not installed. Run: pip install datasets")
        return {}
    
    print(f"\n{'='*80}")
    print(f"MS MARCO Passage Ranking Evaluation")
    print(f"{'='*80}\n")
    
    # Load MS MARCO dataset
    print("Loading MS MARCO dataset...")
    print("Note: This downloads ~3GB of data on first run")
    
    try:
        # Load passages (corpus)
        passages = load_dataset("ms_marco", "v1.1", split="train")
        
        # Load queries and qrels
        queries_dataset = load_dataset("ms_marco", "v1.1", split="validation")
        
        print(f"✓ Loaded {len(passages)} passages")
        print(f"✓ Loaded {len(queries_dataset)} queries")
        
    except Exception as e:
        print(f"✗ Error loading MS MARCO: {e}")
        print("\nAlternative: Use smaller subset")
        return {}
    
    # Build corpus dict (limit to first 10K passages for memory)
    print("\nBuilding corpus (using first 10K passages)...")
    corpus = {}
    max_passages = 10000
    
    for i, passage in enumerate(passages):
        if i >= max_passages:
            break
        
        passage_id = str(passage['passages']['passage_id'][0])
        passage_text = passage['passages']['passage_text'][0]
        
        corpus[passage_id] = {
            'text': passage_text,
            'title': ''
        }
    
    print(f"✓ Built corpus with {len(corpus)} passages")
    
    # Build queries dict (limit for testing)
    print(f"\nBuilding queries (using first {num_queries})...")
    queries = {}
    qrels = defaultdict(dict)
    
    for i, item in enumerate(queries_dataset):
        if i >= num_queries:
            break
        
        query_id = str(item['query_id'])
        query_text = item['query']
        
        queries[query_id] = query_text
        
        # Build qrels (ground truth)
        if 'passages' in item and 'is_selected' in item['passages']:
            for passage_id, is_selected in zip(
                item['passages']['passage_id'],
                item['passages']['is_selected']
            ):
                if is_selected == 1:
                    qrels[query_id][str(passage_id)] = 1
    
    print(f"✓ Built {len(queries)} queries with {sum(len(q) for q in qrels.values())} relevance judgments")
    
    # Initialize Fractal AI Memory Layer
    print("\nInitializing Fractal AI Memory Layer...")
    memory = MemoryLayer(
        session_name='msmarco_eval',
        max_context_tokens=200_000_000,
        target_compression_ratio=0.00001,
        enable_persistence=False,
        device=device
    )
    
    # Ingest corpus
    print("\nIngesting corpus into Fractal AI...")
    corpus_parts = []
    passage_ids = []
    
    for passage_id, passage_data in corpus.items():
        corpus_parts.append(passage_data['text'])
        passage_ids.append(passage_id)
    
    combined_corpus = "\n\n=== PASSAGE SEPARATOR ===\n\n".join(corpus_parts)
    
    start_time = time.time()
    summary = memory.ingest_context(combined_corpus, show_progress=False)
    ingestion_time = time.time() - start_time
    
    print(f"✓ Ingested {summary['tokens_ingested']:,} tokens in {ingestion_time:.1f}s")
    print(f"✓ Extracted {summary['patterns_extracted']:,} patterns")
    
    # Run retrieval
    print(f"\nRunning retrieval on {len(queries)} queries...")
    results = {}
    retrieval_times = []
    
    if RICH_AVAILABLE:
        progress = Progress()
        task = progress.add_task("[cyan]Retrieving...", total=len(queries))
        progress.start()
    
    for query_id, query_text in queries.items():
        start = time.time()
        
        # Query Fractal AI
        memory_result = memory.query(query_text, max_context_tokens=5000)
        
        retrieval_time = (time.time() - start) * 1000
        retrieval_times.append(retrieval_time)
        
        # Map retrieved context to passage IDs
        # Simplified: score passages based on presence in retrieved context
        retrieved_context = memory_result['context'].lower()
        
        passage_scores = {}
        for passage_id in passage_ids:
            passage_text = corpus[passage_id]['text'].lower()
            
            # Simple overlap scoring
            if passage_text[:50] in retrieved_context:
                passage_scores[passage_id] = 1.0
            elif passage_text[:20] in retrieved_context:
                passage_scores[passage_id] = 0.5
            else:
                passage_scores[passage_id] = 0.1
        
        results[query_id] = passage_scores
        
        if RICH_AVAILABLE:
            progress.update(task, advance=1)
    
    if RICH_AVAILABLE:
        progress.stop()
    
    # Calculate metrics
    print("\nCalculating metrics...")
    
    mrr_10 = calculate_mrr(qrels, results, k=10)
    mrr_100 = calculate_mrr(qrels, results, k=100)
    recall_100 = calculate_recall(qrels, results, k=100)
    recall_1000 = calculate_recall(qrels, results, k=1000)
    
    avg_retrieval_time = np.mean(retrieval_times)
    
    # Display results
    print(f"\n{'='*80}")
    print("RESULTS")
    print(f"{'='*80}\n")
    
    if RICH_AVAILABLE:
        table = Table(title="MS MARCO Results", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Score", style="green", justify="right")
        
        table.add_row("MRR@10", f"{mrr_10:.4f}")
        table.add_row("MRR@100", f"{mrr_100:.4f}")
        table.add_row("Recall@100", f"{recall_100:.4f}")
        table.add_row("Recall@1000", f"{recall_1000:.4f}")
        table.add_row("Avg Retrieval Time", f"{avg_retrieval_time:.1f}ms")
        table.add_row("Queries Evaluated", f"{len(queries)}")
        
        console.print(table)
    else:
        print(f"MRR@10:        {mrr_10:.4f}")
        print(f"MRR@100:       {mrr_100:.4f}")
        print(f"Recall@100:    {recall_100:.4f}")
        print(f"Recall@1000:   {recall_1000:.4f}")
        print(f"Avg Retrieval: {avg_retrieval_time:.1f}ms")
        print(f"Queries:       {len(queries)}")
    
    # Compare to baselines
    print(f"\n{'='*80}")
    print("BASELINE COMPARISON")
    print(f"{'='*80}\n")
    
    baselines = {
        'BM25': {'MRR@10': 0.187, 'Recall@1000': 0.853},
        'SBERT': {'MRR@10': 0.330, 'Recall@1000': 0.958},
        'DPR': {'MRR@10': 0.318, 'Recall@1000': 0.945},
        'ColBERT': {'MRR@10': 0.360, 'Recall@1000': 0.968},
    }
    
    print("Published baselines (full dev set):")
    for model, scores in baselines.items():
        print(f"\n{model}:")
        for metric, score in scores.items():
            print(f"  {metric}: {score:.4f}")
    
    print(f"\nFractal AI (subset of {num_queries} queries):")
    print(f"  MRR@10:       {mrr_10:.4f}")
    print(f"  Recall@1000:  {recall_1000:.4f}")
    
    return {
        'mrr@10': mrr_10,
        'mrr@100': mrr_100,
        'recall@100': recall_100,
        'recall@1000': recall_1000,
        'avg_retrieval_ms': avg_retrieval_time,
        'num_queries': len(queries)
    }


if __name__ == "__main__":
    print("="*80)
    print("FRACTAL AI - MS MARCO BENCHMARK")
    print("="*80)
    print("\nThis evaluates Fractal AI on the MS MARCO passage ranking task.")
    print("MS MARCO is the industry standard for passage retrieval.\n")
    
    if not DATASETS_AVAILABLE:
        print("Installing datasets...")
        print("Run: pip install datasets")
        sys.exit(1)
    
    # Start with small subset for testing
    print("Starting with 100 queries (increase for full evaluation)")
    print("Full dev set has 6,980 queries\n")
    
    try:
        metrics = evaluate_msmarco_dev(num_queries=100, device="cuda")
        
        print("\n" + "="*80)
        print("EVALUATION COMPLETE")
        print("="*80)
        print("\nNext steps:")
        print("1. Increase num_queries to 1000+ for more robust evaluation")
        print("2. Improve passage-to-pattern mapping for better accuracy")
        print("3. Compare to baselines on same query subset")
        print("4. Analyze failure cases and improve pattern extraction")
        
    except Exception as e:
        print(f"\n✗ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
