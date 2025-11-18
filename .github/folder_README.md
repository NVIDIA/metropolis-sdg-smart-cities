### Overview

This folder configures contribution workflows for the repository:
- Issue templates (GitHub Issue Forms)
- Pull request template
- Workflow templates
- CODEOWNERS

### Opening Issues

Use the "New issue" button in GitHub and choose the appropriate template:
- Bug Report: Runtime errors, broken behavior, regressions.
- Feature Request: New functionality or enhancements.
- Documentation Request (New/Correction): Missing or incorrect docs.
- Question / Guidance: Usage questions, configuration guidance.

Before filing an issue:
- Search existing issues to avoid duplicates.
- For security vulnerabilities, do not open an issue. Report via NVIDIA PSIRT (see SECURITY.md).

What to include:
- Clear summary and steps to reproduce.
- Environment details (OS, GPU model, driver/CUDA, Docker Engine + Compose versions, NVIDIA Container Toolkit, image tags, GPU assignment).
- Relevant logs and configuration snippets.

### Pull Requests

Use the default PR template. Ensure CI passes and link related issues. Follow the Code of Conduct.

### Workflow Templates

Automation examples for labeling or linking issues to project boards.

### CODEOWNERS

Defines maintainers for code paths; update as teams/ownership evolve.
