# Code Documentation

This document provides an overview of the Fractal AI codebase structure and key components.

##  Project Structure

```
fractal-ai/
├── enhancements/              # Core production code
│   ├── memory_layer.py        # Main memory layer (961 lines)
│   ├── hierarchical_patterns.py
│   ├── selective_compression.py
│   ├── streaming_processor.py
│   ├── persistent_memory.py
│   ├── llm_adapters.py
│   └── examples/
│       └── example_usage.py
│
├── benchmark_beir.py          # BEIR benchmark script
├── benchmark_msmarco.py       # MS MARCO benchmark
└── chat_with_fractal.py       # Interactive interface
```

---

##  Core Components

### 1. Memory Layer (`enhancements/memory_layer.py`)

**Purpose:** Main interface for semantic memory and retrieval.

**Key Classes:**
- `MemoryLayer` - Primary API for ingestion and querying

**Key Methods:**

#### `__init__(session_name, max_context_tokens, device)`
Initializes the memory layer with:
- Sentence transformer model (all-MiniLM-L6-v2, 384D embeddings)
- FAISS index for fast similarity search
- Dynamic dimension adjustment (64-256D based on variance)
- Multi-core CPU + GPU optimization
- Query cache for repeated queries

#### `ingest_context(context, metadata, show_progress)`
Ingests large text documents:
1. Extracts patterns from 6 perspectives (titles, concepts, phrases, evidence, technical terms, general text)
2. Generates embeddings in batch (GPU-accelerated)
3. Applies dynamic dimension reduction (captures 85% variance)
4. Builds/updates FAISS index
5. Stores int8-quantized embeddings (75% memory reduction)

**Returns:** Summary with tokens ingested, patterns extracted, compression ratio

#### `query(query, max_context_tokens, retrieval_strategy)`
Retrieves relevant context for a query:
1. Generates query embedding
2. Checks semantic query cache (>85% similarity = cache hit)
3. FAISS search for top-200 candidate patterns
4. Hybrid reranking: 70% semantic + 30% keyword matching
5. Importance score boosting (titles > concepts > general text)
6. Compresses top-50 patterns into context

**Returns:** Dict with compressed context, token count, patterns used

**Performance:**
- Ingestion: 192 patterns/second
- Query: 10ms average
- nDCG@10: 0.3249 on BEIR scifact

---

### 2. Perspective-Based Extraction

The system analyzes text from 6 semantic perspectives:

#### Perspective 1: Titles/Headings (importance: 1.0)
- Extracts document titles and section headings
- Identifies first few lines, markdown headers (#), short lines (<150 chars)
- Highest importance for structural navigation

#### Perspective 2: Core Concepts (importance: 0.9)
- Extracts key entities and domain concepts
- Uses capitalized phrases and domain-specific terms
- Examples: "Machine Learning", "Neural Network", "Stem Cell"

#### Perspective 3: Key Phrases (importance: 0.8)
- Extracts important multi-word expressions
- Identifies 2-4 word phrases containing important keywords
- Examples: "performance optimization", "accuracy improvement"

#### Perspective 4: Evidence/Data (importance: 0.7)
- Extracts numerical facts and statistics
- Identifies lines with numbers, percentages, measurements
- Examples: "95% accuracy", "10ms latency"

#### Perspective 5: Technical Terms (importance: 0.85)
- Extracts key-value pairs and technical specifications
- Identifies colon-separated patterns (key: value)
- Examples: "API_KEY: xyz", "Model: GPT-4"

#### Perspective 6: General Text (importance: 0.5)
- Extracts general contextual information
- Fallback for comprehensive coverage
- Importance boosted by keyword presence

**Why This Works:**
- Multi-faceted analysis captures semantic richness
- Different perspectives complement each other
- Achieved 268% improvement over single-view baseline (0.0882 → 0.3392 nDCG@10)

---

### 3. Dynamic Dimension Adjustment

**Purpose:** Automatically select optimal embedding dimensionality based on content variance.

**Algorithm:**
1. Compute variance per dimension across all pattern embeddings
2. Sort dimensions by variance (descending)
3. Calculate cumulative variance coverage
4. Select minimum dimensions to capture 85% of variance
5. Clamp to range [64, 256] dimensions

**Benefits:**
- Simple content → 64-128D (homogeneous text)
- Complex content → 200-256D (diverse scientific papers)
- 33% speedup with 98% performance retention
- Adaptive to content characteristics

**Implementation:**
```python
# Compute variance and sort dimensions
variances = np.var(embeddings, axis=0)
sorted_indices = np.argsort(variances)[::-1]

# Find dimensions needed for 85% variance
cumulative_variance = np.cumsum(sorted_variances)
variance_ratios = cumulative_variance / total_variance
dims_needed = np.argmax(variance_ratios >= 0.85) + 1

# Select top dimensions
optimal_dims = np.clip(dims_needed, 64, 256)
dimension_indices = sorted_indices[:optimal_dims]
```

---

### 4. Hybrid Semantic Search

**Purpose:** Combine semantic understanding with keyword precision.

**Scoring Formula:**
```python
# 1. Semantic score from FAISS (cosine similarity)
semantic_score = faiss_distance  # [0, 1]

# 2. Keyword matching score
keyword_score = matched_words / total_query_words  # [0, 1]

# 3. Hybrid combination
combined_score = 0.7 * semantic_score + 0.3 * keyword_score

# 4. Importance boosting
final_score = combined_score * pattern.importance_score
```

**Why 70/30 Split:**
- 70% semantic: Captures deep understanding, handles synonyms, paraphrasing
- 30% keyword: Ensures precision, exact term matches, technical accuracy
- Balances recall (semantic) with precision (keyword)

**Query Cache:**
- Stores recent query results with embeddings
- Semantic matching: reuses results if similarity > 85%
- Provides instant responses for repeated/similar questions
- LRU eviction when cache exceeds 100 entries

---

### 5. FAISS Indexing

**Index Type:** `IndexFlatIP` (Inner Product)

**Why Inner Product:**
- Equivalent to cosine similarity for normalized vectors
- Faster than explicit cosine distance computation
- Exact search (not approximate)

**Normalization:**
```python
# Normalize embeddings before indexing
embedding_norm = embedding / np.linalg.norm(embedding)

# Inner product of normalized vectors = cosine similarity
similarity = np.dot(query_norm, doc_norm)
```

**Search Process:**
1. Normalize query embedding
2. FAISS search returns top-k (distances, indices)
3. Higher distance = more similar (inner product)
4. Retrieve pattern objects from indices
5. Rerank with hybrid scoring

---

### 6. Memory Optimization

**Int8 Quantization:**
```python
# Quantize float32 → int8 (75% memory reduction)
embeddings_int8 = (embeddings_float32 * 127).astype(np.int8)

# Dequantize when needed
embeddings_float32 = embeddings_int8.astype(float32) / 127.0
```

**Benefits:**
- 4 bytes → 1 byte per value
- 75% memory reduction
- Minimal accuracy loss (<1%)
- Enables larger corpora in memory

**Storage:**
- Full embeddings: Quantized int8 in pattern metadata
- Index embeddings: Reduced dimensions (64-256D) in FAISS
- Best of both: Memory efficiency + search speed

---

## 🔬 Benchmark Scripts

### `benchmark_beir.py`

**Purpose:** Evaluate on BEIR scifact dataset (scientific fact verification)

**Dataset:**
- 5,183 documents
- 300 queries
- Task: Retrieve documents supporting/refuting claims

**Process:**
1. Load BEIR scifact dataset
2. Initialize FractalAIRetrieverPhase1
3. Ingest entire corpus (~7MB)
4. Evaluate 300 queries
5. Compute metrics (nDCG@10, Recall@100, MAP, MRR, Precision)

**Expected Results:**
- nDCG@10: 0.3249 (48.9% of SOTA)
- Recall@100: 80.57%
- Ingestion: 96 seconds
- Patterns: 52,316 (10.1 per document)

### `benchmark_msmarco.py`

**Purpose:** Evaluate on MS MARCO passage ranking dataset

**Dataset:**
- Large-scale web search dataset
- Passage retrieval task
- More challenging than scifact

---

##  Key Design Decisions

### 1. Why Perspective-Based Extraction?
- **Problem:** Single-view analysis misses semantic richness
- **Solution:** Analyze from 6 complementary perspectives
- **Result:** 268% improvement over baseline

### 2. Why Dynamic Dimensions?
- **Problem:** Fixed 384D is overkill for simple content, insufficient for complex
- **Solution:** Adaptive 64-256D based on variance analysis
- **Result:** 33% speedup, 98% accuracy retention

### 3. Why Hybrid Search?
- **Problem:** Pure semantic misses exact terms, pure keyword misses semantics
- **Solution:** 70% semantic + 30% keyword combination
- **Result:** Balanced precision and recall

### 4. Why Int8 Quantization?
- **Problem:** Float32 embeddings consume too much memory
- **Solution:** Quantize to int8 (scale by 127)
- **Result:** 75% memory reduction, <1% accuracy loss

### 5. Why FAISS Inner Product?
- **Problem:** Cosine similarity computation is slow
- **Solution:** Normalize vectors, use inner product (equivalent to cosine)
- **Result:** Fast exact search, no approximation needed

---

##  Performance Characteristics

### Ingestion
- **Speed:** 192 patterns/second
- **Time:** 96 seconds for 5,183 documents
- **Patterns:** 10.1 per document average
- **Memory:** Int8 quantized (75% reduction)

### Query
- **Speed:** 10ms average
- **Cache:** Instant for similar queries (>85% similarity)
- **Retrieval:** Top-200 candidates, rerank to top-50
- **Accuracy:** 0.3249 nDCG@10 (48.9% of SOTA)

### Scalability
- **Corpus:** Tested up to 100K+ patterns
- **Dimensions:** Adaptive 64-256D
- **Index:** FAISS exact search (fast even at scale)
- **Memory:** Int8 quantization enables large corpora

---

## 🛠️ Configuration Options

### Memory Layer Initialization
```python
memory = MemoryLayer(
    session_name='my_session',      # Session identifier
    max_context_tokens=100_000,     # Max tokens to store
    target_compression_ratio=0.001, # Compression target (1000:1)
    enable_persistence=True,        # Save/load sessions
    device='cuda'                   # 'cuda' or 'cpu'
)
```

### Dynamic Dimensions
```python
memory.use_dynamic_dimensions = True  # Enable adaptive dims
memory.target_variance_coverage = 0.85  # 85% variance
memory.min_dimensions = 64  # Minimum dims
memory.max_dimensions = 256  # Maximum dims
```

### Multi-Core Optimization
```python
memory.num_workers = 15  # CPU cores to use
memory.use_gpu_batch = True  # GPU batch processing
```

### Query Cache
```python
memory.cache_max_size = 100  # Max cached queries
memory.cache_similarity_threshold = 0.85  # Cache hit threshold
```

---

## 📝 Code Style

- **Docstrings:** All public methods have comprehensive docstrings
- **Comments:** Inline comments explain complex logic
- **Type Hints:** Used throughout for clarity
- **Naming:** Descriptive variable and function names
- **Structure:** Modular design with clear separation of concerns

---

##  Debugging

### Enable Debug Output
The first query automatically prints scoring breakdown:
```
Top 5 pattern scores for query: 'your query'
  1. Combined: 0.856 (Sem: 0.912, Key: 0.667)
     Text: Machine learning enables computers to learn...
```

### Check Statistics
```python
print(memory.stats)
# {
#   'total_tokens_stored': 150000,
#   'total_queries': 42,
#   'total_tokens_sent_to_llm': 8000,
#   'bandwidth_saved_mb': 5.2
# }
```

### Verify FAISS Index
```python
print(f"Patterns indexed: {memory.faiss_index.ntotal}")
print(f"Index dimensions: {memory.adaptive_dim_count}D")
print(f"Variance coverage: {memory._calculate_variance_coverage():.1f}%")
```

---

##  Future Improvements

See [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md) for detailed improvement suggestions:
- Cross-encoder reranking (+10-15%)
- Domain-specific embeddings (+10-15%)
- Query expansion (+5-10%)
- LLM-based extraction (+15-25%)

---

**Last Updated:** April 10, 2026  
**Version:** 1.0.0  
**Status:** Production-ready
