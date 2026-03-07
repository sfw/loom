"""Three-tier verification gates for subtask outputs.

Tier 1: Deterministic checks (free, instant, no LLM)
Tier 2: Independent LLM verification (fresh context, different model)
Tier 3: Voting verification (N independent checks, majority agreement)

The model that performed the work never checks its own output.
"""

from __future__ import annotations

from .gates import VerificationGates, VotingVerifier
from .tier1 import DeterministicVerifier
from .tier2 import LLMVerifier
from .types import Check, VerificationResult

__all__ = [
    "Check",
    "DeterministicVerifier",
    "LLMVerifier",
    "VerificationGates",
    "VerificationResult",
    "VotingVerifier",
]
