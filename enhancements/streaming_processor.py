"""
Streaming and Chunked Processing System

Implements streaming architecture to process 20M+ tokens incrementally without
loading entire context into memory, maintaining CHARM's helical continuity.
"""

import torch
import torch.nn as nn
import math
from typing import List, Dict, Optional, Tuple, Callable, Iterator
from dataclasses import dataclass
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


@dataclass
class ChunkMetadata:
    """Metadata for a processing chunk."""
    chunk_id: int
    start_position: int
    end_position: int
    start_turn: int
    end_turn: int
    is_turn_aligned: bool
    overlap_size: int


class CHARMChunkManager:
    """
    Manages streaming chunks while preserving CHARM's helical structure.
    Ensures smooth transitions between chunks at helix boundaries.
    """
    
    def __init__(
        self,
        chunk_size: int = 256_000,
        overlap: int = 1024,
        helix_diameter: int = 32
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.helix_diameter = helix_diameter
        
        # Track helical state across chunks
        self.current_turn = 0
        self.position_in_turn = 0
    
    def create_chunks(
        self,
        token_stream: List[int],
        align_to_turns: bool = True
    ) -> List[ChunkMetadata]:
        """
        Create overlapping chunks aligned to helix boundaries.
        
        Args:
            token_stream: List of token IDs or iterator
            align_to_turns: Whether to align chunk boundaries to helix turns
            
        Returns:
            List of chunk metadata
        """
        chunks = []
        position = 0
        chunk_id = 0
        total_tokens = len(token_stream)
        
        while position < total_tokens:
            # Calculate chunk boundaries
            chunk_start = position
            chunk_end = min(position + self.chunk_size, total_tokens)
            
            # Align end to helix turn boundary if close and requested
            if align_to_turns:
                turn_boundary = ((chunk_end // self.helix_diameter) + 1) * self.helix_diameter
                if turn_boundary - chunk_end < self.helix_diameter // 4:
                    chunk_end = min(turn_boundary, total_tokens)
            
            # Add overlap from previous chunk
            if position > 0:
                chunk_start = max(0, chunk_start - self.overlap)
            
            # Calculate helix positions
            start_turn = chunk_start // self.helix_diameter
            end_turn = chunk_end // self.helix_diameter
            is_turn_aligned = (chunk_end % self.helix_diameter == 0)
            
            # Create chunk metadata
            chunk = ChunkMetadata(
                chunk_id=chunk_id,
                start_position=chunk_start,
                end_position=chunk_end,
                start_turn=start_turn,
                end_turn=end_turn,
                is_turn_aligned=is_turn_aligned,
                overlap_size=self.overlap if position > 0 else 0
            )
            
            chunks.append(chunk)
            
            # Move to next chunk (accounting for overlap)
            position = chunk_end
            chunk_id += 1
        
        return chunks
    
    def get_chunk_tokens(
        self,
        token_stream: List[int],
        chunk_metadata: ChunkMetadata
    ) -> torch.Tensor:
        """Extract tokens for a specific chunk."""
        tokens = token_stream[chunk_metadata.start_position:chunk_metadata.end_position]
        return torch.tensor(tokens, dtype=torch.long)
    
    def merge_chunk_states(
        self,
        prev_state: torch.Tensor,
        curr_state: torch.Tensor,
        overlap_size: int
    ) -> torch.Tensor:
        """
        Merge hidden states from overlapping regions.
        Uses CHARM's complementary strand attention for smooth blending.
        
        Args:
            prev_state: [prev_seq_len, hidden_size]
            curr_state: [curr_seq_len, hidden_size]
            overlap_size: Number of overlapping tokens
            
        Returns:
            merged_state: Combined hidden states
        """
        if overlap_size == 0:
            return curr_state
        
        # Extract overlap regions
        prev_overlap = prev_state[-overlap_size:]
        curr_overlap = curr_state[:overlap_size]
        
        # Compute helical blend weights
        blend_weights = self._compute_helical_blend_weights(overlap_size)
        
        # Blend overlapping region
        merged_overlap = (
            prev_overlap * blend_weights.unsqueeze(-1) +
            curr_overlap * (1 - blend_weights.unsqueeze(-1))
        )
        
        # Concatenate: prev (without overlap) + merged + curr (without overlap)
        if prev_state.shape[0] > overlap_size:
            result = torch.cat([
                prev_state[:-overlap_size],
                merged_overlap,
                curr_state[overlap_size:]
            ])
        else:
            result = torch.cat([merged_overlap, curr_state[overlap_size:]])
        
        return result
    
    def _compute_helical_blend_weights(self, overlap_size: int) -> torch.Tensor:
        """
        Compute blend weights following helical structure.
        
        Weights transition smoothly from 1.0 (use previous) to 0.0 (use current),
        with modulation based on helix position.
        """
        positions = torch.arange(overlap_size, dtype=torch.float32)
        
        # Helical position within turn
        turn_positions = positions % self.helix_diameter
        
        # Helical modulation (stronger at turn boundaries)
        turn_weights = torch.cos(turn_positions * math.pi / self.helix_diameter)
        turn_weights = (turn_weights + 1) / 2  # Normalize to [0, 1]
        
        # Linear blend from 1 to 0
        linear_weights = torch.linspace(1.0, 0.0, overlap_size)
        
        # Combine: linear trend with helical modulation
        blend_weights = linear_weights * 0.7 + turn_weights * 0.3
        
        return blend_weights


class IncrementalMemoryManager:
    """
    Updates Flat Layered Memory incrementally as chunks are processed.
    Maintains CHARM's recursive memory across chunk boundaries.
    """
    
    def __init__(
        self,
        flat_memory: nn.Module,
        max_memory_layers: int = 24
    ):
        self.flat_memory = flat_memory
        self.max_memory_layers = max_memory_layers
        
        # Persistent memory state
        self.memory_banks = None
        self.total_tokens_processed = 0
        self.tokens_per_layer = 1_000_000  # 1M tokens per layer
    
    def process_chunk(
        self,
        chunk_hidden_states: torch.Tensor,
        chunk_metadata: ChunkMetadata
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process a single chunk and update memory.
        
        Args:
            chunk_hidden_states: [batch_size, seq_len, hidden_size]
            chunk_metadata: Metadata about the chunk
            
        Returns:
            updated_hidden: Processed hidden states
            updated_memory: Updated memory banks
        """
        # Update memory with chunk
        updated_hidden, updated_memory = self.flat_memory(
            hidden_states=chunk_hidden_states,
            memory_banks=self.memory_banks,
            dynamic_growth=True,
            ultra_long_context=True
        )
        
        # Store updated memory for next chunk
        self.memory_banks = updated_memory
        
        # Update token counter
        chunk_size = chunk_metadata.end_position - chunk_metadata.start_position
        self.total_tokens_processed += chunk_size
        
        # Check if we need to add memory layers
        if self._should_add_layer():
            self._add_memory_layer()
        
        return updated_hidden, updated_memory
    
    def _should_add_layer(self) -> bool:
        """Determine if we need to add a new memory layer."""
        expected_layers = min(
            self.total_tokens_processed // self.tokens_per_layer,
            self.max_memory_layers
        )
        current_layers = self.flat_memory.num_layers
        return expected_layers > current_layers
    
    def _add_memory_layer(self):
        """Add a new memory layer dynamically."""
        if self.flat_memory.num_layers < self.max_memory_layers:
            self.flat_memory.num_layers += 1
            self.flat_memory.active_layer_mask[self.flat_memory.num_layers - 1] = True
            print(f"Added memory layer {self.flat_memory.num_layers} at {self.total_tokens_processed:,} tokens")
    
    def get_memory_state(self) -> Dict:
        """Get current memory state information."""
        return {
            'total_tokens_processed': self.total_tokens_processed,
            'num_layers': self.flat_memory.num_layers,
            'memory_banks_shape': self.memory_banks.shape if self.memory_banks is not None else None,
            'expected_layers': min(
                self.total_tokens_processed // self.tokens_per_layer,
                self.max_memory_layers
            )
        }


class StreamingProcessor:
    """
    Main streaming pipeline for processing 20M+ tokens.
    Coordinates chunking, CHARM processing, and memory management.
    """
    
    def __init__(
        self,
        model: nn.Module,
        chunk_size: int = 256_000,
        overlap: int = 1024,
        device: str = 'cuda'
    ):
        self.model = model
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.device = device
        
        # Initialize components
        self.chunk_manager = CHARMChunkManager(
            chunk_size=chunk_size,
            overlap=overlap
        )
        
        self.memory_manager = IncrementalMemoryManager(
            flat_memory=model.flat_memory if hasattr(model, 'flat_memory') else None
        )
        
        # Statistics
        self.stats = {
            'chunks_processed': 0,
            'total_tokens': 0,
            'total_time_seconds': 0.0,
            'avg_tokens_per_second': 0.0
        }
    
    def process_stream(
        self,
        token_stream: List[int],
        progress_callback: Optional[Callable[[int, int, int], None]] = None,
        extract_patterns: bool = True
    ) -> Dict:
        """
        Process token stream in chunks.
        
        Args:
            token_stream: List of token IDs or iterator
            progress_callback: Optional callback(chunk_idx, total_chunks, tokens_processed)
            extract_patterns: Whether to extract patterns during processing
            
        Returns:
            Aggregated results and final memory state
        """
        import time
        
        start_time = time.time()
        
        # Create chunks
        print(f"Creating chunks for {len(token_stream):,} tokens...")
        chunks = self.chunk_manager.create_chunks(token_stream)
        print(f"Created {len(chunks)} chunks")
        
        results = []
        prev_hidden = None
        all_patterns = []
        
        for i, chunk_metadata in enumerate(chunks):
            chunk_start_time = time.time()
            
            # Get chunk tokens
            chunk_tokens = self.chunk_manager.get_chunk_tokens(token_stream, chunk_metadata)
            chunk_tokens = chunk_tokens.unsqueeze(0).to(self.device)  # Add batch dimension
            
            # Process chunk through model
            with torch.no_grad():
                chunk_output = self.model(
                    input_ids=chunk_tokens,
                    ultra_long_context=True
                )
            
            # Get hidden states
            if hasattr(chunk_output, 'last_hidden_state'):
                chunk_hidden = chunk_output.last_hidden_state.squeeze(0)
            else:
                chunk_hidden = chunk_output.squeeze(0)
            
            # Merge with previous chunk if overlap exists
            if prev_hidden is not None and chunk_metadata.overlap_size > 0:
                chunk_hidden = self.chunk_manager.merge_chunk_states(
                    prev_hidden,
                    chunk_hidden,
                    chunk_metadata.overlap_size
                )
            
            # Update memory incrementally
            if self.memory_manager.flat_memory is not None:
                updated_hidden, updated_memory = self.memory_manager.process_chunk(
                    chunk_hidden.unsqueeze(0),  # Add batch dimension
                    chunk_metadata
                )
                chunk_hidden = updated_hidden.squeeze(0)
            
            # Extract patterns if requested
            chunk_patterns = []
            if extract_patterns and hasattr(self.model, 'pattern_extractor'):
                # Convert tokens back to text (simplified - would need tokenizer)
                # For now, skip pattern extraction in streaming mode
                pass
            
            # Store for next iteration
            prev_hidden = chunk_hidden
            
            # Update statistics
            chunk_time = time.time() - chunk_start_time
            chunk_size = chunk_metadata.end_position - chunk_metadata.start_position
            self.stats['chunks_processed'] += 1
            self.stats['total_tokens'] += chunk_size
            self.stats['total_time_seconds'] += chunk_time
            
            # Progress callback
            if progress_callback:
                progress_callback(
                    i + 1,
                    len(chunks),
                    self.memory_manager.total_tokens_processed
                )
            
            # Store results
            results.append({
                'chunk_id': i,
                'chunk_metadata': chunk_metadata,
                'hidden_states': chunk_hidden,
                'patterns': chunk_patterns,
                'processing_time': chunk_time,
                'tokens_per_second': chunk_size / chunk_time if chunk_time > 0 else 0
            })
            
            # Print progress
            if (i + 1) % 10 == 0 or i == len(chunks) - 1:
                elapsed = time.time() - start_time
                tokens_per_sec = self.stats['total_tokens'] / elapsed if elapsed > 0 else 0
                print(f"Processed chunk {i+1}/{len(chunks)} "
                      f"({self.stats['total_tokens']:,} tokens, "
                      f"{tokens_per_sec:.0f} tok/s)")
        
        # Calculate final statistics
        total_time = time.time() - start_time
        self.stats['avg_tokens_per_second'] = (
            self.stats['total_tokens'] / total_time if total_time > 0 else 0
        )
        
        return {
            'chunks_processed': len(chunks),
            'total_tokens': self.stats['total_tokens'],
            'total_time_seconds': total_time,
            'avg_tokens_per_second': self.stats['avg_tokens_per_second'],
            'final_memory': self.memory_manager.memory_banks,
            'memory_state': self.memory_manager.get_memory_state(),
            'all_patterns': all_patterns,
            'results': results
        }
    
    def process_stream_generator(
        self,
        token_iterator: Iterator[int],
        max_tokens: Optional[int] = None
    ) -> Iterator[Dict]:
        """
        Process token stream as generator for very large streams.
        
        Yields chunk results as they are processed.
        """
        buffer = []
        position = 0
        chunk_id = 0
        
        for token in token_iterator:
            buffer.append(token)
            position += 1
            
            # Process when buffer reaches chunk size
            if len(buffer) >= self.chunk_size:
                chunk_metadata = ChunkMetadata(
                    chunk_id=chunk_id,
                    start_position=position - len(buffer),
                    end_position=position,
                    start_turn=(position - len(buffer)) // 32,
                    end_turn=position // 32,
                    is_turn_aligned=(position % 32 == 0),
                    overlap_size=self.overlap if chunk_id > 0 else 0
                )
                
                # Process chunk
                chunk_tokens = torch.tensor(buffer, dtype=torch.long).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    chunk_output = self.model(input_ids=chunk_tokens, ultra_long_context=True)
                
                chunk_hidden = chunk_output.last_hidden_state.squeeze(0) if hasattr(chunk_output, 'last_hidden_state') else chunk_output.squeeze(0)
                
                # Update memory
                if self.memory_manager.flat_memory is not None:
                    updated_hidden, _ = self.memory_manager.process_chunk(
                        chunk_hidden.unsqueeze(0),
                        chunk_metadata
                    )
                
                # Yield result
                yield {
                    'chunk_id': chunk_id,
                    'chunk_metadata': chunk_metadata,
                    'tokens_processed': position
                }
                
                # Keep overlap for next chunk
                buffer = buffer[-self.overlap:] if self.overlap > 0 else []
                chunk_id += 1
            
            # Stop if max tokens reached
            if max_tokens and position >= max_tokens:
                break
        
        # Process remaining buffer
        if buffer:
            chunk_metadata = ChunkMetadata(
                chunk_id=chunk_id,
                start_position=position - len(buffer),
                end_position=position,
                start_turn=(position - len(buffer)) // 32,
                end_turn=position // 32,
                is_turn_aligned=(position % 32 == 0),
                overlap_size=self.overlap if chunk_id > 0 else 0
            )
            
            chunk_tokens = torch.tensor(buffer, dtype=torch.long).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                chunk_output = self.model(input_ids=chunk_tokens, ultra_long_context=True)
            
            yield {
                'chunk_id': chunk_id,
                'chunk_metadata': chunk_metadata,
                'tokens_processed': position
            }


class StreamingPatternAggregator:
    """Aggregates patterns extracted during streaming processing."""
    
    def __init__(self):
        self.patterns = []
        self.pattern_index = {}
    
    def add_patterns(self, patterns: List, chunk_start_position: int):
        """Add patterns from a chunk, adjusting positions."""
        for pattern in patterns:
            # Adjust pattern positions to global coordinates
            if hasattr(pattern, 'start_pos'):
                pattern.start_pos += chunk_start_position
            if hasattr(pattern, 'end_pos'):
                pattern.end_pos += chunk_start_position
            if hasattr(pattern, 'rope_position'):
                pattern.rope_position += chunk_start_position
            
            self.patterns.append(pattern)
            
            # Index by pattern name
            if hasattr(pattern, 'pattern_name'):
                if pattern.pattern_name not in self.pattern_index:
                    self.pattern_index[pattern.pattern_name] = []
                self.pattern_index[pattern.pattern_name].append(pattern)
    
    def get_all_patterns(self) -> List:
        """Get all aggregated patterns."""
        return self.patterns
    
    def get_patterns_by_name(self, pattern_name: str) -> List:
        """Get patterns of a specific type."""
        return self.pattern_index.get(pattern_name, [])
    
    def get_statistics(self) -> Dict:
        """Get pattern statistics."""
        return {
            'total_patterns': len(self.patterns),
            'pattern_types': len(self.pattern_index),
            'patterns_by_type': {
                name: len(patterns)
                for name, patterns in self.pattern_index.items()
            }
        }
