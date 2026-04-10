# Integration Guide: Enhancements with Core CHARM Architecture

This guide explains how to integrate the enhancements package with the core CHARM architecture from `active_architecture/`.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                         │
│              (Your code using the enhancements)              │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                  Enhancements Package                        │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Memory Layer API (memory_layer.py)                  │  │
│  │  - MemoryLayer: Core memory interface                │  │
│  │  - TokenMemoryAgent: High-level agent                │  │
│  └──────────────────────────────────────────────────────┘  │
│                              ↓                               │
│  ┌─────────────┬─────────────┬─────────────┬─────────────┐ │
│  │ Hierarchical│  Selective  │  Streaming  │ Persistent  │ │
│  │  Patterns   │ Compression │  Processor  │   Memory    │ │
│  └─────────────┴─────────────┴─────────────┴─────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│              Core CHARM Architecture                         │
│              (active_architecture/)                          │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  IntegratedFractalArchitecture                       │  │
│  │  - AdaptiveCHARMModel                                │  │
│  │  - FlatLayeredMemory                                 │  │
│  │  - RoPEAwarePatternExtractor                         │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Step-by-Step Integration

### Step 1: Import Required Modules

```python
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Core CHARM architecture
from active_architecture.integrated_architecture import IntegratedFractalArchitecture
from active_architecture.main import create_config

# Enhancements
from enhancements import (
    HierarchicalPatternRetriever,
    StreamingProcessor,
    MemorySession,
    TokenMemoryAgent
)
```

### Step 2: Initialize Core Model

```python
# Create configuration
config = create_config(
    max_position_embeddings=25_000_000,
    max_memory_layers=24,
    memory_growth_factor=1_000_000,
    chunk_size=256_000,
    use_flash_attention=True,
    use_memory_compression=True,
    compression_ratio=0.25
)

# Initialize core CHARM model
model = IntegratedFractalArchitecture(
    config=config,
    max_patterns=100_000,
    device='cuda'
)
```

### Step 3: Add Hierarchical Pattern Retrieval

```python
# Replace basic pattern extractor with hierarchical retriever
hierarchical_retriever = HierarchicalPatternRetriever(
    hot_cache_size=1000,
    recent_tier_size=10000,
    archive_tier_size=100000,
    helix_diameter=config.helix_diameter
)

# Wrap the model's pattern extractor
original_extractor = model.pattern_extractor

def enhanced_extract_patterns(text, hidden_states=None, rope_positions=None):
    # Use original extractor
    patterns = original_extractor.extract_patterns(text, hidden_states, rope_positions)
    
    # Index in hierarchical retriever
    for pattern in patterns:
        hierarchical_retriever.index_pattern(pattern)
    
    return patterns

# Replace extraction method
model.pattern_extractor.extract_patterns = enhanced_extract_patterns
```

### Step 4: Add Streaming Processor

```python
# Initialize streaming processor
streaming_processor = StreamingProcessor(
    model=model,
    chunk_size=256_000,
    overlap=1024,
    device='cuda'
)

# Process large token stream
def process_large_context(tokens, show_progress=True):
    def progress_callback(chunk_idx, total_chunks, tokens_processed):
        if show_progress:
            print(f"Chunk {chunk_idx}/{total_chunks} ({tokens_processed:,} tokens)")
    
    results = streaming_processor.process_stream(
        token_stream=tokens,
        progress_callback=progress_callback,
        extract_patterns=True
    )
    
    return results
```

### Step 5: Add Persistent Memory

```python
# Initialize session manager
session_manager = MemorySession(session_dir='./memory_sessions')

# Save session
def save_current_session(session_name, metadata=None):
    session_manager.save_session(
        session_name=session_name,
        model=model,
        metadata=metadata
    )
    print(f"Session '{session_name}' saved")

# Load session
def load_previous_session(session_name):
    metadata = session_manager.load_session(
        session_name=session_name,
        model=model,
        device='cuda'
    )
    print(f"Session '{session_name}' loaded")
    return metadata
```

### Step 6: Create Unified Interface

```python
class EnhancedFractalAI:
    """
    Unified interface combining core CHARM with all enhancements.
    """
    
    def __init__(self, session_name, config=None):
        self.session_name = session_name
        
        # Initialize core model
        self.config = config or create_config()
        self.model = IntegratedFractalArchitecture(
            config=self.config,
            max_patterns=100_000
        )
        
        # Initialize enhancements
        self.hierarchical_retriever = HierarchicalPatternRetriever(
            hot_cache_size=1000,
            recent_tier_size=10000,
            archive_tier_size=100000
        )
        
        self.streaming_processor = StreamingProcessor(
            model=self.model,
            chunk_size=256_000,
            overlap=1024
        )
        
        self.session_manager = MemorySession()
    
    def ingest(self, tokens, show_progress=True):
        """Ingest large context with streaming."""
        results = self.streaming_processor.process_stream(
            token_stream=tokens,
            progress_callback=lambda c, t, p: print(f"{c}/{t}") if show_progress else None
        )
        return results
    
    def query(self, query_text, strategy='hybrid', top_k=10):
        """Query with hierarchical retrieval."""
        # Get query embedding from model
        query_tokens = self._tokenize(query_text)
        with torch.no_grad():
            query_output = self.model(query_tokens)
            query_embedding = query_output.last_hidden_state.mean(dim=1)
        
        # Retrieve patterns
        patterns = self.hierarchical_retriever.retrieve(
            query=query_text,
            query_embedding=query_embedding,
            retrieval_strategy=strategy,
            top_k=top_k
        )
        
        return patterns
    
    def save(self, metadata=None):
        """Save session."""
        self.session_manager.save_session(
            session_name=self.session_name,
            model=self.model,
            metadata=metadata
        )
    
    def load(self):
        """Load session."""
        return self.session_manager.load_session(
            session_name=self.session_name,
            model=self.model
        )
    
    def _tokenize(self, text):
        """Tokenize text (placeholder)."""
        # Would use actual tokenizer
        return torch.randint(0, 50000, (1, len(text.split())))
```

## Usage Example

```python
# Initialize enhanced system
fractal_ai = EnhancedFractalAI(
    session_name='my_project',
    config=create_config(max_position_embeddings=25_000_000)
)

# Ingest 20M tokens
with open('large_corpus.txt', 'r') as f:
    text = f.read()
    tokens = tokenize(text)  # Your tokenizer

results = fractal_ai.ingest(tokens, show_progress=True)
print(f"Processed {results['total_tokens']:,} tokens")

# Query
patterns = fractal_ai.query(
    "What is the API key?",
    strategy='hybrid',
    top_k=10
)

for pattern in patterns:
    print(f"Found: {pattern.extracted_values}")

# Save session
fractal_ai.save(metadata={
    'corpus': 'legal_documents',
    'date': '2026-04-04'
})

# Later: Load session
fractal_ai.load()
```

## Integration with LLM Adapters

```python
from enhancements import MemoryLayerLLM

# Create memory layer + LLM system
llm_system = MemoryLayerLLM(
    session_name='my_project',
    llm_provider='gpt4',
    llm_api_key=os.getenv('OPENAI_API_KEY'),
    max_context_tokens=4000
)

# Ingest corpus
llm_system.ingest(large_corpus)

# Chat with cost tracking
response = llm_system.chat(
    "What precedents were cited?",
    retrieval_strategy='hybrid'
)

print(f"Response: {response['response']}")
print(f"Cost: ${response['cost_info']['total_cost']:.4f}")
print(f"Saved: ${response['savings']['cost_saved']:.2f}")
```

## Performance Tuning

### For 20M Tokens

```python
config = create_config(
    max_position_embeddings=25_000_000,
    max_memory_layers=24,
    memory_layer_size=256,
    memory_growth_factor=1_000_000,
    chunk_size=256_000,
    compression_ratio=0.25,
    use_flash_attention=True
)
```

### For 100M Tokens

```python
config = create_config(
    max_position_embeddings=150_000_000,
    max_memory_layers=100,
    memory_layer_size=512,
    memory_growth_factor=1_000_000,
    chunk_size=256_000,
    compression_ratio=0.15,
    use_flash_attention=True
)
```

## Testing Integration

```bash
# Run integration test
cd enhancements/examples
python demo_full_integration.py

# Run individual component tests
python test_hierarchical_patterns.py
python test_memory_layer_llm.py
```

## Troubleshooting

### Issue: Out of Memory

**Solution**: Reduce chunk size or enable gradient checkpointing

```python
config.chunk_size = 128_000  # Reduce from 256K
model.gradient_checkpointing_enabled = True
```

### Issue: Slow Pattern Retrieval

**Solution**: Adjust tier sizes or use more aggressive caching

```python
hierarchical_retriever = HierarchicalPatternRetriever(
    hot_cache_size=2000,  # Increase hot cache
    recent_tier_size=20000,
    archive_tier_size=100000
)
```

### Issue: Pattern Database Too Large

**Solution**: Implement pattern pruning

```python
# Prune low-importance patterns
from enhancements.persistent_memory import PatternDatabase

db = PatternDatabase('patterns.db')
low_importance = db.query_by_importance(min_score=0.0, limit=10000)

# Keep only top 50% by importance
# (Implementation would delete bottom 50%)
```

## Next Steps

1. **Run Full Benchmark**: Test with actual 20M token corpus
2. **Optimize Performance**: Profile and optimize bottlenecks
3. **Deploy as Service**: Containerize and deploy
4. **Build Client SDKs**: Python, JavaScript, REST API
5. **Production Hardening**: Error handling, monitoring, logging

## Support

For issues or questions:
- Check `enhancements/README.md` for component details
- Review `ENHANCEMENTS_20M_TOKENS.md` for design rationale
- See `examples/` for usage patterns
