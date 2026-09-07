# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability within ΔSearch, please send an email to **sachncs@gmail.com**. All security vulnerabilities will be promptly addressed.

**Please do not report security vulnerabilities through public GitHub issues.**

### What to include

When reporting a vulnerability, please include:

- A description of the vulnerability
- Steps to reproduce the issue
- Potential impact
- Any suggested fixes (if applicable)

### Response timeline

- **Acknowledgment**: Within 48 hours
- **Initial assessment**: Within 1 week
- **Fix or mitigation**: Depends on severity, typically within 2 weeks
- **Disclosure**: Coordinated with the reporter

## Security Best Practices

When using ΔSearch in production:

- **Input validation**: Always validate graph data before passing to ΔSearch
- **Memory limits**: Be aware that large graphs can consume significant memory
- **Thread safety**: Use `ThreadSafeGraph` for concurrent access
- **Dependency updates**: Keep your Python environment up to date

## Dependencies

This project has **zero runtime dependencies** — it uses only the Python standard library. This significantly reduces the attack surface.

For development dependencies, see `pyproject.toml`.

## Security Hardening

- All user inputs are validated at the API boundary
- Self-loops are rejected to prevent undefined behavior
- Defensive copies prevent accidental state mutation
- The undo-stack pattern avoids catastrophic deepcopy operations

## Questions?

If you have questions about security practices, feel free to open a discussion or reach out to the maintainers.
