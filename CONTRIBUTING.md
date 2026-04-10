# Contributing to Fractal AI

Thank you for your interest in contributing to Fractal AI! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/yourusername/fractal-ai.git`
3. Create a branch: `git checkout -b feature/your-feature-name`
4. Install dependencies: `pip install -r requirements.txt`

## Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt

# Run tests (if available)
pytest

# Format code
black enhancements/
```

## Code Style

- Follow PEP 8 guidelines
- Use type hints where appropriate
- Write docstrings for all public functions
- Keep functions focused and modular

## Testing

Before submitting a PR:
1. Test your changes on a small dataset
2. Verify no performance regression
3. Ensure code passes linting

## Benchmark Testing

To verify performance:

```bash
# Run BEIR benchmark
python benchmark_beir.py

# Expected: nDCG@10 >= 0.32
```

## Pull Request Process

1. Update documentation if needed
2. Add tests for new features
3. Ensure all tests pass
4. Update CHANGELOG.md
5. Submit PR with clear description

## Areas for Contribution

### High Priority
- Cross-encoder reranking implementation
- Domain-specific embedding fine-tuning
- Query expansion features
- Additional benchmark datasets

### Medium Priority
- Performance optimizations
- Documentation improvements
- Example notebooks
- Additional language support

### Low Priority
- UI improvements
- Visualization tools
- Alternative embedding models

## Performance Guidelines

Contributions should maintain or improve:
- **nDCG@10:** >= 0.32 on BEIR scifact
- **Speed:** <= 100s for 5K documents
- **Memory:** No significant increase

## Questions?

Open an issue for:
- Bug reports
- Feature requests
- Documentation clarifications
- General questions

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
