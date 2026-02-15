"""Process definition system: declarative domain specialization.

Process definitions are YAML files (or directories) that specialize
Loom's generic orchestration engine for specific domains — marketing
strategy, financial analysis, research reports, etc.

The engine reads a process definition and injects domain intelligence
(persona, phase blueprints, tool guidance, verification rules, memory
guidance) into its existing prompt templates. No Python code per domain.

Supports both single-file processes (``*.yaml``) and directory-based
process packages that can bundle tools, templates, and examples::

    # Single file
    ~/.loom/processes/quick-research.yaml

    # Package directory
    ~/.loom/processes/financial-analysis/
    ├── process.yaml
    ├── tools/
    │   └── dcf_builder.py
    └── examples/
        └── equity-analysis.yaml
"""

from loom.processes.schema import (
    ProcessDefinition,
    ProcessLoader,
    ProcessNotFoundError,
    ProcessValidationError,
)

__all__ = [
    "ProcessDefinition",
    "ProcessLoader",
    "ProcessNotFoundError",
    "ProcessValidationError",
]
