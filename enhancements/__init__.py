"""
Fractal AI Enhancements for 20M+ Token Memory Agent

This package contains enhancements to scale the CHARM architecture to 20M+ tokens:
1. Hierarchical pattern storage and retrieval
2. Selective compression based on content importance
3. Streaming/chunked processing with helical continuity
4. Persistent memory system for cross-session storage
5. Memory layer API for commercial LLM integration
"""

__version__ = "1.0.0"
__author__ = "Fractal AI Team"

from .hierarchical_patterns import (
    CHARMPatternCluster,
    HierarchicalPatternRetriever,
    SemanticPatternIndex
)

from .selective_compression import (
    CHARMImportanceScorer,
    AdaptiveCompressionLayer,
    ContentClassifier
)

from .streaming_processor import (
    CHARMChunkManager,
    IncrementalMemoryManager,
    StreamingProcessor
)

from .persistent_memory import (
    MemorySerializer,
    PatternDatabase,
    MemorySession
)

from .memory_layer import (
    MemoryLayer,
    TokenMemoryAgent
)

from .llm_adapters import (
    LLMAdapter,
    GPT4Adapter,
    ClaudeAdapter,
    GeminiAdapter,
    MemoryLayerLLM
)

__all__ = [
    # Hierarchical patterns
    'CHARMPatternCluster',
    'HierarchicalPatternRetriever',
    'SemanticPatternIndex',
    
    # Selective compression
    'CHARMImportanceScorer',
    'AdaptiveCompressionLayer',
    'ContentClassifier',
    
    # Streaming
    'CHARMChunkManager',
    'IncrementalMemoryManager',
    'StreamingProcessor',
    
    # Persistence
    'MemorySerializer',
    'PatternDatabase',
    'MemorySession',
    
    # Memory layer
    'MemoryLayer',
    'TokenMemoryAgent',
    
    # LLM adapters
    'LLMAdapter',
    'GPT4Adapter',
    'ClaudeAdapter',
    'GeminiAdapter',
    'MemoryLayerLLM',
]
