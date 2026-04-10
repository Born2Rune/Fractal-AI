# Changelog

All notable changes to Fractal AI will be documented in this file.

## [1.0.0] - 2026-04-10

### Production Release

**Performance:** 0.3249 nDCG@10 on BEIR scifact (48.9% of SOTA)

### Added
- Perspective-based pattern extraction (6 semantic perspectives)
- Dynamic dimension adjustment (adaptive 64-256D embeddings)
- Multi-core CPU + GPU optimization
- Hybrid semantic search (70% semantic + 30% keyword)
- Int8 quantization for memory efficiency
- FAISS inner product indexing
- BEIR benchmark integration
- MS MARCO benchmark integration
- Comprehensive documentation

### Performance Improvements
- 77x improvement over baseline (0.0042 → 0.3249 nDCG@10)
- 192 patterns/second ingestion speed
- 10ms average query time
- 96 seconds for 5,183 documents
- 75% memory reduction with Int8 quantization

### Optimizations Tested and Removed
- Phase 2: Variance-based dimension selection (regressed -19%)
- Phase 3: Prominence scoring (regressed -4.5%)
- Diffusion clustering (no improvement, high overhead)
- Proposition extraction (25-50x slower, impractical)

### Documentation
- Comprehensive README.md
- Detailed benchmark results
- Production system documentation
- Project structure guide
- MIT License

---

## Development History

### [0.9.0] - Phase 3 Testing
- Tested prominence scoring (regressed performance)
- Tested diffusion clustering (no benefit)
- Removed both optimizations

### [0.8.0] - Phase 2 Testing
- Tested variance-based dimension selection
- Found regression in performance (-19%)
- Removed optimization

### [0.7.0] - Phase 1 Implementation
- Implemented perspective-based extraction
- Achieved 0.3392 nDCG@10 (+268% improvement)
- Became foundation for production system

### [0.6.0] - Dynamic Dimensions
- Added adaptive dimension selection
- 33% speedup with 98% performance retention
- Integrated with Phase 1

### [0.5.0] - Multi-Core Optimization
- Added parallel processing
- GPU batch embedding generation
- 192 patterns/second ingestion

### [0.4.0] - Semantic Search
- Integrated sentence-transformers
- Hybrid scoring implementation
- FAISS indexing

### [0.3.0] - Pattern Extraction
- Basic pattern extraction
- Hierarchical storage
- CHARM importance scoring

### [0.2.0] - BEIR Integration
- Fixed zero-score bug
- Document-to-pattern mapping
- Baseline: 0.0882 nDCG@10

### [0.1.0] - Initial Implementation
- Basic memory layer
- Original implementation (broken)
- nDCG@10: 0.0042

---

**Status:** Production-ready (v1.0.0)  
**Last Updated:** April 10, 2026
