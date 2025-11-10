# Contributing to Dual-Attention Whisper

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## How to Contribute

### Reporting Bugs

If you find a bug, please create an issue with:
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, GPU, Python version, library versions)
- Error messages or logs

### Suggesting Enhancements

We welcome suggestions for:
- New features
- Performance improvements
- Documentation improvements
- Better training strategies

Please create an issue with:
- Clear description of the enhancement
- Rationale for why it would be useful
- Potential implementation approach (if applicable)

### Pull Requests

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/your-feature-name`
3. **Make your changes**:
   - Follow existing code style
   - Add tests if applicable
   - Update documentation
4. **Test your changes**:
   - Run existing tests
   - Test on sample data
   - Check memory usage and performance
5. **Commit your changes**: Use clear, descriptive commit messages
6. **Push to your fork**: `git push origin feature/your-feature-name`
7. **Create a Pull Request**: Describe your changes and link any related issues

## Code Style

- Follow PEP 8 for Python code
- Use type hints where applicable
- Add docstrings to functions and classes
- Keep functions focused and reasonably sized
- Use meaningful variable names

## Testing Guidelines

- Test with different model sizes (tiny, small, medium)
- Test with different batch sizes
- Verify memory usage on target GPUs
- Check training convergence
- Validate inference output quality

## Documentation

When adding features, please update:
- README.md (if it affects quick start or main features)
- USAGE_GUIDE.md (detailed usage instructions)
- CHANGELOG.md (list your changes)
- Code comments and docstrings

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/DualAttention.git
cd DualAttention

# Install in development mode
pip install -e .

# Install development dependencies
pip install pytest black flake8
```

## Questions?

Feel free to open an issue for questions or join discussions.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
