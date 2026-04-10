"""
Fractal AI - BEIR Benchmark Evaluation

Benchmarks the Fractal AI memory layer on the BEIR (Benchmarking Information Retrieval)
framework using the scifact dataset (scientific fact verification).

This script evaluates:
- Perspective-based pattern extraction (Phase 1)
- Dynamic dimension adjustment
- Hybrid semantic + keyword search
- Overall retrieval quality (nDCG@10, Recall@100, MAP, etc.)

Dataset: BEIR scifact
- 5,183 scientific documents
- 300 queries
- Task: Retrieve documents that support/refute scientific claims

Expected Performance:
- nDCG@10: ~0.32 (48.9% of SOTA)
- Recall@100: ~80%
- Ingestion: ~96 seconds

Usage:
    python benchmark_beir.py
"""

import sys
from pathlib import Path
import time
import numpy as np
from collections import defaultdict
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent))

from beir import util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval

from enhancements.memory_layer import MemoryLayer


class FractalAIRetrieverPhase1:
    """
    BEIR-compatible retriever using Fractal AI memory layer.
    
    Implements the BEIR retrieval interface with:
    - Perspective-based pattern extraction (6 perspectives)
    - Dynamic dimension adjustment (64-256D)
    - Hybrid semantic + keyword search (70/30 split)
    - FAISS vector indexing
    
    The retriever ingests the entire corpus on initialization and provides
    a search() method for query evaluation.
    """
    
    def __init__(self, corpus: Dict[str, Dict[str, str]], **kwargs):
        """
        Initialize retriever and ingest corpus.
        
        Args:
            corpus: Dict mapping doc_id -> {'title': str, 'text': str}
            **kwargs: Additional arguments (unused, for BEIR compatibility)
        """
        self.corpus = corpus
        
        # Initialize Fractal AI memory layer with production settings
        self.memory = MemoryLayer(
            session_name='beir_phase1',
            max_context_tokens=100_000_000,  # Large enough for full corpus (~7MB)
            target_compression_ratio=0.0001,  # High compression (10,000:1)
            enable_persistence=False,  # No need to save for benchmark
            device='cuda'  # Use GPU for fast embedding generation
        )
        
        # Ingest entire corpus (extracts patterns, generates embeddings, builds FAISS index)
        self._ingest_corpus()
    
    def _ingest_corpus(self):
        """
        Ingest corpus into memory layer.
        
        Combines all documents into a single text and ingests it. The memory layer
        will extract patterns from 6 perspectives, generate embeddings, and build
        a FAISS index for fast retrieval.
        
        Also stores document texts for later pattern-to-document mapping during search.
        """
        print("Ingesting corpus with Phase 1 perspective-based extraction...")
        
        # Store lowercase document texts for pattern matching during search
        # This allows us to map retrieved patterns back to source documents
        self.doc_texts = {}
        all_doc_texts = []
        
        # Process each document: combine title and text
        for doc_id, doc in self.corpus.items():
            title = doc.get('title', '')
            text = doc.get('text', '')
            doc_text = f"{title}\n{text}" if title else text
            
            # Store lowercase version for case-insensitive matching
            self.doc_texts[doc_id] = doc_text.lower()
            all_doc_texts.append((doc_id, doc_text))
        
        # Combine all documents into single corpus text
        # Document separators help the pattern extractor identify boundaries
        combined_corpus = "\n\n=== DOCUMENT SEPARATOR ===\n\n".join(
            [text for _, text in all_doc_texts]
        )
        
        print(f"Ingesting {len(combined_corpus)/1024/1024:.1f} MB corpus...")
        start_time = time.time()
        
        # Ingest into memory layer:
        # 1. Extracts patterns from 6 perspectives (titles, concepts, phrases, etc.)
        # 2. Generates embeddings for all patterns (batch GPU processing)
        # 3. Applies dynamic dimension reduction (384D → 64-256D)
        # 4. Builds FAISS index for fast similarity search
        summary = self.memory.ingest_context(combined_corpus, show_progress=False)
        
        ingestion_time = time.time() - start_time
        print(f"✓ Ingested {summary['tokens_ingested']:,} tokens in {ingestion_time:.1f}s")
        print(f"✓ Extracted {summary['patterns_extracted']:,} patterns")
        print(f"✓ Average {summary['patterns_extracted'] / len(self.corpus):.1f} patterns per document")
    
    def search(
        self,
        corpus: Dict[str, Dict[str, str]],
        queries: Dict[str, str],
        top_k: int,
        **kwargs
    ) -> Dict[str, Dict[str, float]]:
        """
        Search corpus for relevant documents given queries.
        
        For each query:
        1. Generate query embedding
        2. FAISS search for top-k similar patterns
        3. Hybrid reranking (70% semantic + 30% keyword)
        4. Map patterns back to source documents
        5. Aggregate scores per document
        
        Args:
            corpus: Document corpus (same as __init__)
            queries: Dict mapping query_id -> query_text
            top_k: Number of documents to retrieve per query
            **kwargs: Additional arguments (unused)
            
        Returns:
            Dict mapping query_id -> {doc_id: score}
        """
        results = {}
        
        print(f"\nEvaluating {len(queries)} queries with Phase 1 extraction...")
        
        for query_idx, (query_id, query_text) in enumerate(queries.items()):
            # Query memory layer using hybrid retrieval:
            # - FAISS semantic search (finds similar patterns)
            # - Keyword matching (ensures term overlap)
            # - Importance boosting (prioritizes titles/concepts)
            memory_result = self.memory.query(
                query_text,
                max_context_tokens=10000,  # Retrieve substantial context
                retrieval_strategy='hybrid'  # Use hybrid semantic + keyword
            )
            
            # Get retrieved context (compressed patterns)
            retrieved_context = memory_result['context'].lower()
            
            # Map retrieved patterns back to source documents
            # Score each document by how much of its content appears in retrieved context
            doc_scores = {}
            
            # Check each document for overlap with retrieved context
            for doc_id in corpus.keys():
                doc_text = self.doc_texts[doc_id]
                
                # Use sentence-level matching for better accuracy
                doc_sentences = [s.strip() for s in doc_text.split('.') if len(s.strip()) > 20]
                
                if not doc_sentences:
                    doc_scores[doc_id] = 0.0
                    continue
                
                # Count how many doc sentences appear in retrieved context
                matches = 0
                for sent in doc_sentences[:10]:  # Check first 10 sentences
                    # Check if substantial part of sentence appears
                    if len(sent) > 30:
                        # Check first and last parts
                        if sent[:30] in retrieved_context or sent[-30:] in retrieved_context:
                            matches += 1
                    elif sent in retrieved_context:
                        matches += 1
                
                # Score based on match ratio
                if matches > 0:
                    doc_scores[doc_id] = matches / min(len(doc_sentences), 10)
                else:
                    doc_scores[doc_id] = 0.0
            
            # Sort by score and take top_k
            sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
            results[query_id] = {doc_id: score for doc_id, score in sorted_docs[:top_k] if score > 0}
            
            # Add small scores for remaining docs (BEIR expects all docs ranked)
            if len(results[query_id]) < top_k:
                remaining = top_k - len(results[query_id])
                for doc_id, score in sorted_docs[len(results[query_id]):len(results[query_id]) + remaining]:
                    if doc_id not in results[query_id]:
                        results[query_id][doc_id] = 0.001
            
            # Progress indicator
            if (query_idx + 1) % 50 == 0:
                print(f"  Processed {query_idx + 1}/{len(queries)} queries...")
        
        return results


def evaluate_on_beir_dataset(
    dataset_name: str = 'scifact',
    split: str = 'test'
):
    """Evaluate Fractal AI on BEIR dataset."""
    
    print("="*80)
    print(f"BEIR Evaluation (PHASE 1): {dataset_name}")
    print("="*80)
    
    # Load dataset
    print(f"\nLoading {dataset_name} dataset...")
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
    data_path = util.download_and_unzip(url, "datasets")
    
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=split)
    
    print(f"✓ Loaded {len(corpus)} documents, {len(queries)} queries")
    
    # Initialize retriever
    print(f"Initializing Fractal AI with {len(corpus)} documents...")
    retriever = FractalAIRetrieverPhase1(corpus)
    
    # Run evaluation
    print("\nRunning evaluation...")
    start_time = time.time()
    
    results = retriever.search(corpus, queries, top_k=100)
    
    eval_time = time.time() - start_time
    print(f"\n✓ Evaluation complete in {eval_time:.1f}s")
    
    # Evaluate results
    ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(qrels, results, [1, 3, 5, 10, 100])
    mrr = EvaluateRetrieval.evaluate_custom(qrels, results, [10], metric="mrr")
    
    # Print results
    print(f"\n{'='*80}")
    print(f"BEIR Results (PHASE 1): {dataset_name}")
    print(f"{'='*80}\n")
    
    # Format results table
    from rich.console import Console
    from rich.table import Table
    
    console = Console()
    table = Table(title=f"BEIR Results (PHASE 1): {dataset_name}")
    
    table.add_column("Metric", style="cyan")
    table.add_column("Fractal AI", style="green")
    table.add_column("Baseline", style="yellow")
    table.add_column("Δ", style="magenta")
    
    # Baselines from BEIR paper
    baselines = {
        'scifact': {'nDCG@10': 0.665, 'Recall@100': 0.908},
        'nfcorpus': {'nDCG@10': 0.325, 'Recall@100': 0.284}
    }
    
    baseline = baselines.get(dataset_name, {})
    
    # Add rows
    metrics = [
        ('nDCG@10', ndcg['NDCG@10'], baseline.get('nDCG@10')),
        ('nDCG@100', ndcg['NDCG@100'], None),
        ('MAP@10', _map['MAP@10'], None),
        ('Recall@10', recall['Recall@10'], None),
        ('Recall@100', recall['Recall@100'], baseline.get('Recall@100')),
        ('MRR@10', mrr['MRR@10'], None),
        ('Precision@10', precision['P@10'], None),
    ]
    
    for metric_name, value, baseline_val in metrics:
        baseline_str = f"{baseline_val:.4f}" if baseline_val else ""
        delta_str = f"{value - baseline_val:.4f}" if baseline_val else ""
        table.add_row(metric_name, f"{value:.4f}", baseline_str, delta_str)
    
    console.print(table)
    
    # Analysis
    print(f"\n{'='*80}")
    print("ANALYSIS")
    print(f"{'='*80}\n")
    
    print(f"Fractal AI nDCG@10: {ndcg['NDCG@10']:.4f}")
    print(f"Previous (fixed):   0.0882")
    print(f"Improvement:        {((ndcg['NDCG@10'] / 0.0882) - 1) * 100:+.1f}%")
    
    if baseline.get('nDCG@10'):
        print(f"\nBaseline (SBERT):   {baseline['nDCG@10']:.4f}")
        print(f"Gap:                {(1 - ndcg['NDCG@10'] / baseline['nDCG@10']) * 100:.1f}%")
    
    # Interpretation
    if ndcg['NDCG@10'] >= 0.20:
        print("\n✓ Good performance - Phase 1 successful!")
    elif ndcg['NDCG@10'] >= 0.10:
        print("\n⚠ Moderate improvement - needs Phase 2")
    else:
        print("\n✗ Low performance - needs investigation")
    
    return {
        'ndcg': ndcg,
        'map': _map,
        'recall': recall,
        'precision': precision,
        'mrr': mrr
    }


if __name__ == '__main__':
    # Run on scifact
    results = evaluate_on_beir_dataset('scifact')
    
    print(f"\n{'='*80}")
    print("PHASE 1 BENCHMARK COMPLETE")
    print(f"{'='*80}")
