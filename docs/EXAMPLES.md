# Example Usage

This document provides examples of how to use Fractal AI in different scenarios.

## Basic Usage

### Simple Document Ingestion and Query

```python
from enhancements.memory_layer import MemoryLayer

# Initialize memory layer
memory = MemoryLayer(
    session_name='my_documents',
    max_context_tokens=100_000,
    device='cuda'  # or 'cpu'
)

# Ingest a document
document = """
Quantum computing represents a paradigm shift in computational power.
Unlike classical computers that use bits (0 or 1), quantum computers
use qubits that can exist in superposition states. This enables
exponentially faster processing for certain types of problems.
"""

summary = memory.ingest_context(document)
print(f"Extracted {summary['patterns_extracted']} patterns")
print(f"Compression ratio: {summary['compression_ratio']:.2%}")

# Query the memory
result = memory.query("What is quantum computing?", max_context_tokens=500)
print(result['context'])
```

## Advanced Configuration

### High-Performance Setup

```python
from enhancements.memory_layer import MemoryLayer

# Configure for maximum performance
memory = MemoryLayer(
    session_name='high_performance',
    max_context_tokens=1_000_000,
    target_compression_ratio=0.0001,  # 10,000:1 compression
    enable_persistence=True,
    device='cuda'
)

# Optimize settings
memory.use_dynamic_dimensions = True
memory.target_variance_coverage = 0.85  # 85% variance
memory.min_dimensions = 64
memory.max_dimensions = 256
memory.num_workers = 15  # Use all CPU cores
memory.use_gpu_batch = True  # GPU batch processing
```

### Memory-Optimized Setup

```python
# Configure for low memory usage
memory = MemoryLayer(
    session_name='memory_optimized',
    max_context_tokens=100_000,
    device='cpu'
)

# Use smaller dimensions
memory.use_dynamic_dimensions = True
memory.max_dimensions = 128  # Reduce max dimensions
memory.target_variance_coverage = 0.75  # Lower variance target
```

## Use Cases

### 1. Scientific Literature Review

```python
memory = MemoryLayer(session_name='literature_review', device='cuda')

# Ingest multiple papers
papers = [
    "paper1.txt",
    "paper2.txt",
    "paper3.txt"
]

for paper_file in papers:
    with open(paper_file, 'r') as f:
        text = f.read()
    memory.ingest_context(text)

# Query across all papers
queries = [
    "What are the main findings about neural networks?",
    "How do the papers compare different approaches?",
    "What future work is suggested?"
]

for query in queries:
    result = memory.query(query, max_context_tokens=1000)
    print(f"\nQuery: {query}")
    print(f"Answer: {result['context']}\n")
```

### 2. Code Documentation Search

```python
memory = MemoryLayer(session_name='code_docs', device='cuda')

# Ingest codebase documentation
import os
for root, dirs, files in os.walk('docs/'):
    for file in files:
        if file.endswith('.md'):
            with open(os.path.join(root, file), 'r') as f:
                memory.ingest_context(f.read())

# Search for specific functionality
result = memory.query(
    "How do I implement custom pattern extraction?",
    max_context_tokens=2000
)
print(result['context'])
```

### 3. Conversation History Management

```python
memory = MemoryLayer(
    session_name='conversation_history',
    enable_persistence=True,
    device='cuda'
)

# Store conversation turns
conversation = [
    "User: What is machine learning?",
    "AI: Machine learning is a subset of artificial intelligence...",
    "User: How does it differ from deep learning?",
    "AI: Deep learning is a specialized form of machine learning..."
]

for turn in conversation:
    memory.ingest_context(turn)

# Query conversation history
result = memory.query(
    "What did we discuss about deep learning?",
    max_context_tokens=500
)
print(result['context'])
```

## Integration with LLMs

### Using with OpenAI API

```python
from enhancements.memory_layer import MemoryLayer
import openai

memory = MemoryLayer(session_name='openai_integration', device='cuda')

# Ingest large context
with open('large_document.txt', 'r') as f:
    memory.ingest_context(f.read())

# Query and send to OpenAI
user_question = "Summarize the key points about climate change"
context = memory.query(user_question, max_context_tokens=3000)

response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Context: {context['context']}\n\nQuestion: {user_question}"}
    ]
)

print(response.choices[0].message.content)
```

### Using with Local LLM (Llama)

```python
from enhancements.memory_layer import MemoryLayer
from transformers import AutoTokenizer, AutoModelForCausalLM

memory = MemoryLayer(session_name='llama_integration', device='cuda')
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

# Ingest documents
memory.ingest_context(large_text)

# Query and generate
user_query = "What are the main themes?"
context = memory.query(user_query, max_context_tokens=2000)

prompt = f"Context: {context['context']}\n\nQuestion: {user_query}\nAnswer:"
inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
outputs = model.generate(**inputs, max_length=500)
print(tokenizer.decode(outputs[0]))
```

## Performance Monitoring

```python
import time

memory = MemoryLayer(session_name='performance_test', device='cuda')

# Measure ingestion time
start = time.time()
summary = memory.ingest_context(large_document)
ingestion_time = time.time() - start

print(f"Ingestion time: {ingestion_time:.2f}s")
print(f"Patterns extracted: {summary['patterns_extracted']}")
print(f"Patterns per second: {summary['patterns_extracted'] / ingestion_time:.1f}")

# Measure query time
start = time.time()
result = memory.query("test query", max_context_tokens=1000)
query_time = time.time() - start

print(f"Query time: {query_time*1000:.2f}ms")
```

## Error Handling

```python
from enhancements.memory_layer import MemoryLayer

try:
    memory = MemoryLayer(
        session_name='error_handling',
        device='cuda'
    )
    
    # Attempt ingestion
    summary = memory.ingest_context(document)
    
    # Attempt query
    result = memory.query(query, max_context_tokens=1000)
    
except RuntimeError as e:
    print(f"Runtime error: {e}")
    # Fallback to CPU
    memory = MemoryLayer(session_name='error_handling', device='cpu')
    
except Exception as e:
    print(f"Unexpected error: {e}")
```

## See Also

- [README.md](../README.md) - Main documentation
- [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md) - Performance metrics
- [FINAL_PRODUCTION_SYSTEM.md](FINAL_PRODUCTION_SYSTEM.md) - System architecture
