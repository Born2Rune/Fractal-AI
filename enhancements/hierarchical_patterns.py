"""
Hierarchical Pattern Storage and Retrieval System

Implements a multi-tier indexing system for 100K+ patterns with CHARM-aware clustering.
"""

import torch
import torch.nn as nn
import math
import numpy as np
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass
from collections import OrderedDict
import hashlib
import time


@dataclass
class PatternMatch:
    """Pattern match with CHARM-aware positioning."""
    pattern_name: str
    start_pos: int
    end_pos: int
    rope_position: int
    extracted_values: Dict
    metadata: Dict
    hidden_states: Optional[torch.Tensor] = None
    importance_score: float = 0.5
    helix_turn: int = 0
    position_in_turn: int = 0
    
    def __post_init__(self):
        """Calculate helix position after initialization."""
        helix_diameter = 32
        self.helix_turn = self.rope_position // helix_diameter
        self.position_in_turn = self.rope_position % helix_diameter


class CHARMPatternCluster:
    """
    Clusters patterns based on helical position and semantic similarity.
    Maintains CHARM's complementary strand relationships.
    """
    
    def __init__(self, helix_diameter: int = 32):
        self.helix_diameter = helix_diameter
        self.clusters: Dict[Tuple[int, int], List[PatternMatch]] = {}
        self.strand_pairs: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
        
    def add_pattern(self, pattern: PatternMatch) -> Tuple[int, int]:
        """
        Add pattern to appropriate cluster based on helix position.
        
        Returns:
            Cluster key (turn_number, cluster_id)
        """
        # Calculate helix position
        turn_number = pattern.rope_position // self.helix_diameter
        position_in_turn = pattern.rope_position % self.helix_diameter
        
        # Find complementary position (opposite side of helix)
        complementary_pos = (position_in_turn + self.helix_diameter // 2) % self.helix_diameter
        
        # Cluster by turn and position (8 clusters per turn)
        cluster_id = position_in_turn // 4
        cluster_key = (turn_number, cluster_id)
        
        if cluster_key not in self.clusters:
            self.clusters[cluster_key] = []
        
        self.clusters[cluster_key].append(pattern)
        
        # Track complementary relationships
        comp_cluster_id = complementary_pos // 4
        comp_key = (turn_number, comp_cluster_id)
        
        if comp_key not in self.strand_pairs:
            self.strand_pairs[comp_key] = []
        if cluster_key not in self.strand_pairs[comp_key]:
            self.strand_pairs[comp_key].append(cluster_key)
        
        return cluster_key
    
    def get_cluster(self, turn_number: int, cluster_id: int) -> List[PatternMatch]:
        """Get patterns in a specific cluster."""
        return self.clusters.get((turn_number, cluster_id), [])
    
    def get_complementary_clusters(self, turn_number: int, cluster_id: int) -> List[List[PatternMatch]]:
        """Get patterns from complementary strand positions."""
        cluster_key = (turn_number, cluster_id)
        complementary_keys = self.strand_pairs.get(cluster_key, [])
        return [self.clusters.get(key, []) for key in complementary_keys]
    
    def get_turn_range(self, start_turn: int, end_turn: int) -> List[PatternMatch]:
        """Get all patterns within a range of helix turns."""
        patterns = []
        for turn in range(start_turn, end_turn + 1):
            for cluster_id in range(8):  # 8 clusters per turn
                patterns.extend(self.get_cluster(turn, cluster_id))
        return patterns


class SemanticPatternIndex:
    """
    Uses CHARM's hidden states for semantic similarity search.
    Maintains helical organization for locality-sensitive hashing.
    """
    
    def __init__(self, hidden_size: int = 768, num_hash_tables: int = 8):
        self.hidden_size = hidden_size
        self.num_hash_tables = num_hash_tables
        
        # Helical hash functions (rotated projections)
        self.hash_functions = []
        self.hash_tables: List[Dict[int, Set[int]]] = []
        
        for i in range(num_hash_tables):
            # Rotate hash projection by helix angle
            angle = (2 * math.pi * i) / num_hash_tables
            rotation_matrix = self._create_rotation_matrix(angle, hidden_size)
            self.hash_functions.append(rotation_matrix)
            self.hash_tables.append({})
        
        # Pattern storage
        self.patterns: Dict[int, PatternMatch] = {}
        self.pattern_embeddings: Dict[int, torch.Tensor] = {}
        self.next_id = 0
    
    def _create_rotation_matrix(self, angle: float, dim: int) -> torch.Tensor:
        """Create rotation matrix for helical hashing."""
        # Create random projection matrix with helical rotation
        matrix = torch.randn(dim, dim // 4)
        
        # Apply rotation in 2D subspaces
        for i in range(0, dim - 1, 2):
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)
            if i < matrix.shape[0] - 1:
                # Rotate pairs of dimensions
                matrix[i] = matrix[i] * cos_a
                matrix[i+1] = matrix[i+1] * sin_a
        
        return matrix
    
    def _hash(self, embedding: torch.Tensor, hash_fn: torch.Tensor) -> int:
        """Compute hash value using helical projection."""
        if embedding.dim() > 1:
            embedding = embedding.flatten()
        
        # Ensure same dtype
        embedding = embedding.float().cpu()
        
        # Project and hash
        projection = torch.matmul(embedding, hash_fn)
        # Use sign bits as hash
        hash_val = int(''.join(['1' if x > 0 else '0' for x in projection[:32]]), 2)
        return hash_val
    
    def add_pattern(self, pattern: PatternMatch) -> int:
        """Add pattern to semantic index."""
        pattern_id = self.next_id
        self.next_id += 1
        
        self.patterns[pattern_id] = pattern
        
        if pattern.hidden_states is not None:
            embedding = pattern.hidden_states
            self.pattern_embeddings[pattern_id] = embedding
            
            # Add to all hash tables
            for i, hash_fn in enumerate(self.hash_functions):
                hash_val = self._hash(embedding, hash_fn)
                if hash_val not in self.hash_tables[i]:
                    self.hash_tables[i][hash_val] = set()
                self.hash_tables[i][hash_val].add(pattern_id)
        
        return pattern_id
    
    def query_similar(self, query_embedding: torch.Tensor, top_k: int = 10) -> List[Tuple[PatternMatch, float]]:
        """Find top-k similar patterns using helical LSH."""
        if query_embedding is None:
            return []
        
        # Query all hash tables
        candidates = set()
        for i, hash_fn in enumerate(self.hash_functions):
            hash_val = self._hash(query_embedding, hash_fn)
            candidates.update(self.hash_tables[i].get(hash_val, set()))
        
        # Rank by actual similarity
        ranked = self._rank_by_similarity(query_embedding, candidates)
        return ranked[:top_k]
    
    def _rank_by_similarity(self, query: torch.Tensor, candidate_ids: Set[int]) -> List[Tuple[PatternMatch, float]]:
        """Rank candidates by cosine similarity."""
        results = []
        query_flat = query.flatten()
        
        for pattern_id in candidate_ids:
            if pattern_id in self.pattern_embeddings:
                embedding = self.pattern_embeddings[pattern_id].flatten()
                similarity = torch.cosine_similarity(
                    query_flat.unsqueeze(0),
                    embedding.unsqueeze(0)
                ).item()
                results.append((self.patterns[pattern_id], similarity))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results


class HierarchicalPatternRetriever:
    """
    Unified interface for multi-tier pattern retrieval.
    Maintains CHARM principles across all tiers.
    
    Three-tier system:
    - L0: Hot cache (1K patterns, <1ms)
    - L1: Recent tier (10K patterns, <10ms)
    - L2: Archive tier (100K patterns, <100ms)
    """
    
    def __init__(
        self,
        hot_cache_size: int = 1000,
        recent_tier_size: int = 10000,
        archive_tier_size: int = 100000,
        helix_diameter: int = 32
    ):
        self.hot_cache_size = hot_cache_size
        self.recent_tier_size = recent_tier_size
        self.archive_tier_size = archive_tier_size
        
        # L0: Hot cache (LRU)
        self.hot_cache: OrderedDict[str, PatternMatch] = OrderedDict()
        
        # L1: Recent tier (CHARM clustering + semantic index)
        self.recent_cluster = CHARMPatternCluster(helix_diameter)
        self.recent_semantic = SemanticPatternIndex()
        self.recent_patterns: Dict[str, PatternMatch] = {}
        
        # L2: Archive tier (compressed storage)
        self.archive_cluster = CHARMPatternCluster(helix_diameter)
        self.archive_semantic = SemanticPatternIndex()
        self.archive_patterns: Dict[str, PatternMatch] = {}
        
        # Statistics
        self.stats = {
            'l0_hits': 0,
            'l1_hits': 0,
            'l2_hits': 0,
            'total_queries': 0,
            'avg_retrieval_time_ms': 0.0
        }
    
    def _pattern_key(self, pattern: PatternMatch) -> str:
        """Generate unique key for pattern."""
        key_str = f"{pattern.pattern_name}:{pattern.start_pos}:{pattern.end_pos}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def index_pattern(self, pattern: PatternMatch):
        """Index a pattern into appropriate tier."""
        pattern_key = self._pattern_key(pattern)
        
        # Add to L0 hot cache
        if pattern_key in self.hot_cache:
            # Move to end (most recent)
            self.hot_cache.move_to_end(pattern_key)
        else:
            self.hot_cache[pattern_key] = pattern
            
            # Evict if cache full
            if len(self.hot_cache) > self.hot_cache_size:
                # Move oldest to L1
                old_key, old_pattern = self.hot_cache.popitem(last=False)
                self._move_to_recent_tier(old_key, old_pattern)
    
    def _move_to_recent_tier(self, pattern_key: str, pattern: PatternMatch):
        """Move pattern from L0 to L1."""
        self.recent_patterns[pattern_key] = pattern
        self.recent_cluster.add_pattern(pattern)
        self.recent_semantic.add_pattern(pattern)
        
        # Evict to L2 if L1 full
        if len(self.recent_patterns) > self.recent_tier_size:
            # Move oldest to L2 (simple FIFO for now)
            old_key = next(iter(self.recent_patterns))
            old_pattern = self.recent_patterns.pop(old_key)
            self._move_to_archive(old_key, old_pattern)
    
    def _move_to_archive(self, pattern_key: str, pattern: PatternMatch):
        """Move pattern from L1 to L2."""
        # Compress hidden states for archive
        if pattern.hidden_states is not None:
            # Simple compression: store in FP16
            pattern.hidden_states = pattern.hidden_states.half()
        
        self.archive_patterns[pattern_key] = pattern
        self.archive_cluster.add_pattern(pattern)
        self.archive_semantic.add_pattern(pattern)
        
        # Evict oldest if archive full
        if len(self.archive_patterns) > self.archive_tier_size:
            old_key = next(iter(self.archive_patterns))
            self.archive_patterns.pop(old_key)
    
    def retrieve(
        self,
        query: str,
        retrieval_strategy: str = 'hybrid',
        query_embedding: Optional[torch.Tensor] = None,
        rope_position: Optional[int] = None,
        top_k: int = 10
    ) -> List[PatternMatch]:
        """
        Retrieve patterns using specified strategy.
        
        Strategies:
        - 'exact': Hash-based exact match
        - 'semantic': Similarity-based search
        - 'temporal': RoPE position-based range query
        - 'hybrid': Combination of above
        - 'complementary': Search complementary strand positions
        """
        start_time = time.time()
        self.stats['total_queries'] += 1
        
        results = []
        
        if retrieval_strategy == 'exact':
            results = self._retrieve_exact(query)
        elif retrieval_strategy == 'semantic':
            results = self._retrieve_semantic(query_embedding, top_k)
        elif retrieval_strategy == 'temporal':
            results = self._retrieve_temporal(rope_position, top_k)
        elif retrieval_strategy == 'complementary':
            results = self._retrieve_complementary(rope_position, top_k)
        elif retrieval_strategy == 'hybrid':
            results = self._retrieve_hybrid(query, query_embedding, rope_position, top_k)
        
        # Update statistics
        elapsed_ms = (time.time() - start_time) * 1000
        self.stats['avg_retrieval_time_ms'] = (
            (self.stats['avg_retrieval_time_ms'] * (self.stats['total_queries'] - 1) + elapsed_ms) /
            self.stats['total_queries']
        )
        
        return results
    
    def _retrieve_exact(self, query: str) -> List[PatternMatch]:
        """Exact match retrieval."""
        # Search in hot cache first
        for pattern in self.hot_cache.values():
            if query in str(pattern.extracted_values):
                self.stats['l0_hits'] += 1
                return [pattern]
        
        # Search in recent tier
        for pattern in self.recent_patterns.values():
            if query in str(pattern.extracted_values):
                self.stats['l1_hits'] += 1
                return [pattern]
        
        # Search in archive
        for pattern in self.archive_patterns.values():
            if query in str(pattern.extracted_values):
                self.stats['l2_hits'] += 1
                return [pattern]
        
        return []
    
    def _retrieve_semantic(self, query_embedding: Optional[torch.Tensor], top_k: int) -> List[PatternMatch]:
        """Semantic similarity retrieval."""
        if query_embedding is None:
            return []
        
        # Try L0 first (simple linear search for small cache)
        l0_results = []
        for pattern in self.hot_cache.values():
            if pattern.hidden_states is not None:
                similarity = torch.cosine_similarity(
                    query_embedding.flatten().unsqueeze(0),
                    pattern.hidden_states.flatten().unsqueeze(0)
                ).item()
                l0_results.append((pattern, similarity))
        
        if l0_results:
            l0_results.sort(key=lambda x: x[1], reverse=True)
            if l0_results[0][1] > 0.9:  # High similarity threshold
                self.stats['l0_hits'] += 1
                return [p for p, _ in l0_results[:top_k]]
        
        # Try L1 semantic index
        l1_results = self.recent_semantic.query_similar(query_embedding, top_k)
        if l1_results and l1_results[0][1] > 0.7:
            self.stats['l1_hits'] += 1
            return [p for p, _ in l1_results]
        
        # Fall back to L2 archive
        l2_results = self.archive_semantic.query_similar(query_embedding, top_k)
        self.stats['l2_hits'] += 1
        return [p for p, _ in l2_results]
    
    def _retrieve_temporal(self, rope_position: Optional[int], top_k: int) -> List[PatternMatch]:
        """Temporal/position-based retrieval."""
        if rope_position is None:
            return []
        
        helix_diameter = 32
        turn_number = rope_position // helix_diameter
        
        # Search nearby turns (±2 turns)
        results = []
        
        # L1 recent tier
        for turn in range(max(0, turn_number - 2), turn_number + 3):
            results.extend(self.recent_cluster.get_turn_range(turn, turn))
        
        if results:
            self.stats['l1_hits'] += 1
            return results[:top_k]
        
        # L2 archive
        for turn in range(max(0, turn_number - 2), turn_number + 3):
            results.extend(self.archive_cluster.get_turn_range(turn, turn))
        
        self.stats['l2_hits'] += 1
        return results[:top_k]
    
    def _retrieve_complementary(self, rope_position: Optional[int], top_k: int) -> List[PatternMatch]:
        """Retrieve from complementary strand positions."""
        if rope_position is None:
            return []
        
        helix_diameter = 32
        turn_number = rope_position // helix_diameter
        position_in_turn = rope_position % helix_diameter
        cluster_id = position_in_turn // 4
        
        # Get complementary clusters
        results = []
        
        # L1 recent tier
        comp_clusters = self.recent_cluster.get_complementary_clusters(turn_number, cluster_id)
        for cluster in comp_clusters:
            results.extend(cluster)
        
        if results:
            self.stats['l1_hits'] += 1
            return results[:top_k]
        
        # L2 archive
        comp_clusters = self.archive_cluster.get_complementary_clusters(turn_number, cluster_id)
        for cluster in comp_clusters:
            results.extend(cluster)
        
        self.stats['l2_hits'] += 1
        return results[:top_k]
    
    def _retrieve_hybrid(
        self,
        query: str,
        query_embedding: Optional[torch.Tensor],
        rope_position: Optional[int],
        top_k: int
    ) -> List[PatternMatch]:
        """Hybrid retrieval combining multiple strategies."""
        # Try exact match first (fastest)
        exact_results = self._retrieve_exact(query)
        if exact_results:
            return exact_results
        
        # Try semantic if embedding available
        if query_embedding is not None:
            semantic_results = self._retrieve_semantic(query_embedding, top_k)
            if semantic_results:
                return semantic_results
        
        # Fall back to temporal if position available
        if rope_position is not None:
            temporal_results = self._retrieve_temporal(rope_position, top_k)
            if temporal_results:
                return temporal_results
        
        return []
    
    def get_all_patterns(self) -> List[PatternMatch]:
        """Get all indexed patterns."""
        all_patterns = []
        all_patterns.extend(self.hot_cache.values())
        all_patterns.extend(self.recent_patterns.values())
        all_patterns.extend(self.archive_patterns.values())
        return all_patterns
    
    def get_statistics(self) -> Dict:
        """Get retrieval statistics."""
        total_hits = self.stats['l0_hits'] + self.stats['l1_hits'] + self.stats['l2_hits']
        
        return {
            **self.stats,
            'total_patterns': len(self.hot_cache) + len(self.recent_patterns) + len(self.archive_patterns),
            'l0_size': len(self.hot_cache),
            'l1_size': len(self.recent_patterns),
            'l2_size': len(self.archive_patterns),
            'l0_hit_rate': self.stats['l0_hits'] / total_hits if total_hits > 0 else 0,
            'l1_hit_rate': self.stats['l1_hits'] / total_hits if total_hits > 0 else 0,
            'l2_hit_rate': self.stats['l2_hits'] / total_hits if total_hits > 0 else 0,
        }
