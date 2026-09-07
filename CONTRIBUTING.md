# Contributing to ΔSearch

Thank you for considering a contribution to ΔSearch! This document outlines the process and guidelines for contributing to this project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Branch Naming](#branch-naming)
- [Commit Messages](#commit-messages)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you agree to uphold its standards.

## Getting Started

1. **Fork** the repository on GitHub
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/delta-search.git
   cd delta-search
   ```
3. **Install** development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```
4. **Create a branch** for your changes:
   ```bash
   git checkout -b feat/my-feature
   ```

## Development Workflow

1. Make your changes on a feature branch
2. Write or update tests as needed
3. Run the full test suite to verify nothing is broken
4. Update documentation if your change affects the public API
5. Submit a pull request

## Branch Naming

Use the following prefixes for branch names:

| Prefix | Purpose |
|--------|---------|
| `feat/` | New features |
| `fix/` | Bug fixes |
| `docs/` | Documentation changes |
| `refactor/` | Code refactoring |
| `test/` | Adding or updating tests |
| `chore/` | Maintenance tasks |
| `perf/` | Performance improvements |

Example: `feat/add-networkx-interop`

## Commit Messages

This project follows [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/):

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

### Types

| Type | Description |
|------|-------------|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation changes |
| `style` | Code style changes (formatting, etc.) |
| `refactor` | Code refactoring |
| `test` | Adding or updating tests |
| `chore` | Maintenance tasks |
| `perf` | Performance improvements |

### Examples

```
feat(graph): add adjacency matrix export method
fix(problem): prevent self-loop edge creation
docs(readme): update installation instructions
test(graph): add edge case tests for empty graphs
```

## Pull Request Process

1. **Update your fork** to match the latest main branch
2. **Ensure all checks pass** (CI, lint, type check, tests)
3. **Fill out the PR template** completely
4. **Request a review** from a maintainer
5. **Address review feedback** promptly
6. **Merge** after approval (maintainers will merge)

### PR Requirements

- [ ] All CI checks pass
- [ ] Tests added for new functionality
- [ ] Documentation updated if applicable
- [ ] No breaking changes without discussion
- [ ] Follows coding standards

## Coding Standards

### Style

- Use **ruff** for linting and formatting
- Target **line length**: 100 characters
- Use **type hints** for all public functions and methods
- Follow **PEP 8** conventions

### Type Checking

- All code must pass `mypy --strict`
- Use `typing` module for all generic types
- Prefer `Protocol` over `Any` for structural typing

### Testing

- Write tests for all new functionality
- Maintain or improve code coverage
- Use descriptive test names
- Follow the Arrange-Act-Assert pattern

### Documentation

- Add docstrings to all public classes and methods
- Use Google-style docstrings
- Include type hints in docstrings when helpful

## Testing

```bash
# Run the full test suite
pytest

# Run with coverage
pytest --cov=delta_search --cov-report=term-missing

# Run specific tests
pytest tests/test_graph.py -v

# Run tests matching a pattern
pytest -k "test_add" -v
```

## Documentation

- Update README.md for user-facing changes
- Add entries to CHANGELOG.md
- Write or update docstrings
- Consider adding examples for new features

## Questions?

If you have questions about contributing, feel free to open a discussion or reach out to the maintainers.
