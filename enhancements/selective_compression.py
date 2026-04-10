"""
Selective Compression Based on Content Importance

Implements adaptive compression that preserves important content while aggressively
compressing filler, maintaining CHARM's helical structure.
"""

import torch
import torch.nn as nn
import math
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


class CHARMImportanceScorer:
    """
    Scores content importance using CHARM's helical attention patterns.
    High attention = high importance.
    """
    
    def __init__(self, helix_diameter: int = 32):
        self.helix_diameter = helix_diameter
        
    def score_segment(
        self,
        hidden_states: torch.Tensor,
        attention_weights: torch.Tensor,
        rope_positions: torch.Tensor,
        pattern_matches: Optional[List] = None
    ) -> torch.Tensor:
        """
        Score importance based on:
        1. Attention received (global attention mask)
        2. Pattern matches (from RoPE-aware extractor)
        3. Helical position (turns get different weights)
        4. Complementary strand activation
        
        Args:
            hidden_states: [seq_len, hidden_size]
            attention_weights: [seq_len] or [seq_len, seq_len]
            rope_positions: [seq_len]
            pattern_matches: List of pattern positions
            
        Returns:
            importance_scores: [seq_len]
        """
        seq_len = hidden_states.shape[0]
        scores = torch.zeros(seq_len)
        
        # Base score from attention
        if attention_weights.dim() == 2:
            # Average attention received from all positions
            base_scores = attention_weights.mean(dim=0)
        else:
            base_scores = attention_weights
        
        for i in range(seq_len):
            # Start with attention score
            score = base_scores[i].item()
            
            # Boost for pattern-matched tokens
            if pattern_matches and self._is_pattern_token(rope_positions[i].item(), pattern_matches):
                score *= 2.0
            
            # Helical position weighting (recent turns more important)
            turn_number = rope_positions[i].item() // self.helix_diameter
            turn_weight = 1.0 + (0.1 * math.log(turn_number + 1))
            
            # Complementary strand bonus
            if self._has_complementary_activation(i, rope_positions, attention_weights):
                score *= 1.5
            
            scores[i] = score * turn_weight
        
        # Normalize to [0, 1]
        if scores.max() > 0:
            scores = scores / scores.max()
        
        return scores
    
    def _is_pattern_token(self, position: int, pattern_matches: List) -> bool:
        """Check if position is part of a pattern match."""
        for pattern in pattern_matches:
            if hasattr(pattern, 'start_pos') and hasattr(pattern, 'end_pos'):
                if pattern.start_pos <= position <= pattern.end_pos:
                    return True
        return False
    
    def _has_complementary_activation(
        self,
        position: int,
        rope_positions: torch.Tensor,
        attention_weights: torch.Tensor,
        threshold: float = 0.5
    ) -> bool:
        """Check if complementary strand position is also active."""
        current_pos = rope_positions[position].item()
        position_in_turn = current_pos % self.helix_diameter
        complementary_pos_in_turn = (position_in_turn + self.helix_diameter // 2) % self.helix_diameter
        
        # Find complementary position in same turn
        turn_number = current_pos // self.helix_diameter
        complementary_pos = turn_number * self.helix_diameter + complementary_pos_in_turn
        
        # Check if complementary position exists and has high attention
        for i, pos in enumerate(rope_positions):
            if pos.item() == complementary_pos:
                if attention_weights.dim() == 2:
                    return attention_weights[i].mean() > threshold
                else:
                    return attention_weights[i] > threshold
        
        return False


class AdaptiveCompressionLayer(nn.Module):
    """
    Applies variable compression based on importance scores.
    Maintains CHARM's recursive memory structure.
    """
    
    def __init__(self, hidden_size: int = 768):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Three compression levels
        self.critical_compressor = nn.Linear(hidden_size, int(hidden_size * 0.9))  # 10% compression
        self.important_compressor = nn.Linear(hidden_size, int(hidden_size * 0.7))  # 30% compression
        self.filler_compressor = nn.Linear(hidden_size, int(hidden_size * 0.1))     # 90% compression
        
        # Decompressors
        self.critical_decompressor = nn.Linear(int(hidden_size * 0.9), hidden_size)
        self.important_decompressor = nn.Linear(int(hidden_size * 0.7), hidden_size)
        self.filler_decompressor = nn.Linear(int(hidden_size * 0.1), hidden_size)
        
        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(hidden_size)
    
    def compress(
        self,
        hidden_states: torch.Tensor,
        importance_scores: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[int]]:
        """
        Apply selective compression.
        
        Args:
            hidden_states: [seq_len, hidden_size]
            importance_scores: [seq_len]
            
        Returns:
            compressed: List of compressed tensors
            compression_map: List indicating which compressor was used (0=critical, 1=important, 2=filler)
        """
        compressed = []
        compression_map = []
        
        for i, (hidden, score) in enumerate(zip(hidden_states, importance_scores)):
            if score > 0.8:  # Critical
                compressed.append(self.critical_compressor(hidden))
                compression_map.append(0)
            elif score > 0.4:  # Important
                compressed.append(self.important_compressor(hidden))
                compression_map.append(1)
            else:  # Filler
                compressed.append(self.filler_compressor(hidden))
                compression_map.append(2)
        
        return compressed, compression_map
    
    def decompress(
        self,
        compressed_states: List[torch.Tensor],
        compression_map: List[int]
    ) -> torch.Tensor:
        """
        Decompress using appropriate decompressor.
        
        Args:
            compressed_states: List of compressed tensors
            compression_map: List indicating which decompressor to use
            
        Returns:
            decompressed: [seq_len, hidden_size]
        """
        decompressed = []
        
        for compressed, comp_type in zip(compressed_states, compression_map):
            if comp_type == 0:
                decompressed.append(self.critical_decompressor(compressed))
            elif comp_type == 1:
                decompressed.append(self.important_decompressor(compressed))
            else:
                decompressed.append(self.filler_decompressor(compressed))
        
        decompressed_tensor = torch.stack(decompressed)
        return self.layer_norm(decompressed_tensor)
    
    def get_compression_ratio(self, compression_map: List[int]) -> float:
        """Calculate effective compression ratio."""
        total_size = 0
        compressed_size = 0
        
        for comp_type in compression_map:
            total_size += self.hidden_size
            if comp_type == 0:
                compressed_size += int(self.hidden_size * 0.9)
            elif comp_type == 1:
                compressed_size += int(self.hidden_size * 0.7)
            else:
                compressed_size += int(self.hidden_size * 0.1)
        
        return compressed_size / total_size if total_size > 0 else 0.0


class ContentClassifier:
    """
    Classifies content into importance tiers.
    Uses CHARM's pattern extraction for classification.
    """
    
    CRITICAL_PATTERNS = [
        'key_value',           # Explicit facts
        'named_entity',        # Important entities
        'question_answer',     # Q&A pairs
        'json_structure',      # Structured data
        'code_block',          # Code
        'api_key',             # Credentials
        'date_value',          # Important dates
    ]
    
    IMPORTANT_PATTERNS = [
        'hierarchical',        # Relationships
        'contextual_key_value', # Contextual facts
        'list_item',           # List elements
        'heading',             # Section headers
    ]
    
    def __init__(self):
        self.critical_keywords = {
            'api', 'key', 'password', 'token', 'secret', 'credential',
            'error', 'warning', 'critical', 'important', 'note',
            'todo', 'fixme', 'bug', 'issue'
        }
        
        self.filler_patterns = {
            'lorem ipsum', 'placeholder', 'example', 'sample',
            'test data', 'dummy', 'filler'
        }
    
    def classify_segment(
        self,
        text: str,
        patterns: Optional[List] = None,
        attention_weights: Optional[torch.Tensor] = None
    ) -> str:
        """
        Classify segment importance.
        
        Returns:
            'critical', 'important', or 'filler'
        """
        text_lower = text.lower()
        
        # Check for filler patterns first
        if any(filler in text_lower for filler in self.filler_patterns):
            return 'filler'
        
        # Check for critical patterns
        if patterns:
            for pattern in patterns:
                if hasattr(pattern, 'pattern_name'):
                    if pattern.pattern_name in self.CRITICAL_PATTERNS:
                        return 'critical'
        
        # Check for critical keywords
        if any(keyword in text_lower for keyword in self.critical_keywords):
            return 'critical'
        
        # Check for important patterns
        if patterns:
            for pattern in patterns:
                if hasattr(pattern, 'pattern_name'):
                    if pattern.pattern_name in self.IMPORTANT_PATTERNS:
                        return 'important'
        
        # Check attention weights
        if attention_weights is not None:
            avg_attention = attention_weights.mean().item()
            if avg_attention > 0.6:
                return 'important'
            elif avg_attention < 0.2:
                return 'filler'
        
        # Default to important (conservative)
        return 'important'
    
    def classify_batch(
        self,
        texts: List[str],
        patterns_list: Optional[List[List]] = None,
        attention_weights_list: Optional[List[torch.Tensor]] = None
    ) -> List[str]:
        """Classify multiple segments."""
        classifications = []
        
        for i, text in enumerate(texts):
            patterns = patterns_list[i] if patterns_list else None
            attention = attention_weights_list[i] if attention_weights_list else None
            classifications.append(self.classify_segment(text, patterns, attention))
        
        return classifications


class CompressionStatistics:
    """Track compression statistics and effectiveness."""
    
    def __init__(self):
        self.total_tokens = 0
        self.critical_tokens = 0
        self.important_tokens = 0
        self.filler_tokens = 0
        self.total_original_size = 0
        self.total_compressed_size = 0
    
    def update(self, compression_map: List[int], hidden_size: int = 768):
        """Update statistics with new compression."""
        for comp_type in compression_map:
            self.total_tokens += 1
            self.total_original_size += hidden_size
            
            if comp_type == 0:  # Critical
                self.critical_tokens += 1
                self.total_compressed_size += int(hidden_size * 0.9)
            elif comp_type == 1:  # Important
                self.important_tokens += 1
                self.total_compressed_size += int(hidden_size * 0.7)
            else:  # Filler
                self.filler_tokens += 1
                self.total_compressed_size += int(hidden_size * 0.1)
    
    def get_statistics(self) -> Dict:
        """Get compression statistics."""
        if self.total_tokens == 0:
            return {
                'total_tokens': 0,
                'compression_ratio': 0.0,
                'space_saved_mb': 0.0
            }
        
        compression_ratio = self.total_compressed_size / self.total_original_size
        space_saved = (self.total_original_size - self.total_compressed_size) * 4 / (1024 * 1024)  # 4 bytes per float, convert to MB
        
        return {
            'total_tokens': self.total_tokens,
            'critical_tokens': self.critical_tokens,
            'important_tokens': self.important_tokens,
            'filler_tokens': self.filler_tokens,
            'critical_percentage': (self.critical_tokens / self.total_tokens) * 100,
            'important_percentage': (self.important_tokens / self.total_tokens) * 100,
            'filler_percentage': (self.filler_tokens / self.total_tokens) * 100,
            'compression_ratio': compression_ratio,
            'space_saved_mb': space_saved,
            'original_size_mb': self.total_original_size * 4 / (1024 * 1024),
            'compressed_size_mb': self.total_compressed_size * 4 / (1024 * 1024)
        }
    
    def reset(self):
        """Reset statistics."""
        self.total_tokens = 0
        self.critical_tokens = 0
        self.important_tokens = 0
        self.filler_tokens = 0
        self.total_original_size = 0
        self.total_compressed_size = 0
