# Spec 06: Verification Gates

## Overview

Verification is a separate system concern in Loom. The model that performed the work never checks its own output. Instead, verification runs through three independent tiers of increasing cost and rigor. This is the single biggest reliability improvement over standard agentic systems.

## Three-Tier Architecture

```
Subtask completes
       │
       ▼
┌─ TIER 1: Deterministic ──────────────────────────┐
│ Free. Instant. No LLM needed.                    │
│ - Output exists?                                  │
│ - Valid JSON/YAML/SQL syntax?                     │
│ - Expected file types present?                    │
│ - Type checks pass?                               │
│ - File sizes non-zero?                            │
│ - Return code == 0?                               │
│ Result: PASS → continue | FAIL → retry            │
└──────────────────────┬────────────────────────────┘
                       │ (if tier >= 2)
                       ▼
┌─ TIER 2: LLM Self-Verification ──────────────────┐
│ One LLM call. Uses DIFFERENT model instance.      │
│ Fresh context: no prior conversation history.     │
│ Input: output + acceptance criteria               │
│ Prompt: "Does this meet criteria? What's wrong?"  │
│ Result: PASS → continue | FAIL → retry with       │
│         feedback from verifier                     │
└──────────────────────┬────────────────────────────┘
                       │ (if tier >= 3)
                       ▼
┌─ TIER 3: Voting Verification ────────────────────┐
│ N independent runs. Compare outputs.              │
│ Expensive — opt-in for critical steps only.       │
│ Run subtask N times independently.                │
│ Compare outputs for consistency.                  │
│ Majority agreement → PASS                         │
│ Divergence → flag for human review                │
│ Principle: "dumb model checked 10x > genius       │
│  model checked once"                              │
└───────────────────────────────────────────────────┘
```

## Verification Gate Orchestrator

```python
class VerificationGates:
    def __init__(
        self,
        model_router: ModelRouter,
        config: VerificationConfig,
    ):
        self._tier1 = DeterministicVerifier()
        self._tier2 = LLMVerifier(model_router)
        self._tier3 = VotingVerifier(model_router)
        self._config = config

    async def verify(
        self,
        subtask: Subtask,
        result: str,
        tool_calls: list[ToolCallRecord],
        workspace: Path | None,
        tier: int = 1,
    ) -> VerificationResult:
        """
        Run verification up to the specified tier.
        Each tier's failure triggers retry before moving to next tier.
        """
        # Tier 1 always runs
        t1 = await self._tier1.verify(subtask, result, tool_calls, workspace)
        if not t1.passed:
            return t1

        if tier < 2 or not self._config.tier2_enabled:
            return t1

        # Tier 2: independent LLM check
        t2 = await self._tier2.verify(subtask, result, tool_calls, workspace)
        if not t2.passed:
            return t2

        if tier < 3 or not self._config.tier3_enabled:
            return t2

        # Tier 3: voting
        t3 = await self._tier3.verify(subtask, result, tool_calls, workspace)
        return t3
```

## Tier 1: Deterministic Verification

```python
class DeterministicVerifier:
    """
    Zero-cost checks that don't require any LLM invocation.
    """

    async def verify(self, subtask, result, tool_calls, workspace) -> VerificationResult:
        checks = []

        # 1. Did tool calls succeed?
        for tc in tool_calls:
            checks.append(Check(
                name=f"tool_{tc.tool}_success",
                passed=tc.result.success,
                detail=tc.result.error if not tc.result.success else None,
            ))

        # 2. Were expected files created/modified?
        if subtask.expected_outputs:
            for expected in subtask.expected_outputs:
                path = workspace / expected if workspace else Path(expected)
                exists = path.exists()
                checks.append(Check(
                    name=f"file_exists_{expected}",
                    passed=exists,
                    detail=f"Expected output '{expected}' not found" if not exists else None,
                ))

        # 3. Non-empty outputs?
        if workspace:
            for tc in tool_calls:
                for f in tc.result.files_changed:
                    path = workspace / f
                    if path.exists():
                        checks.append(Check(
                            name=f"file_nonempty_{f}",
                            passed=path.stat().st_size > 0,
                        ))

        # 4. Valid syntax (if applicable)
        for tc in tool_calls:
            for f in tc.result.files_changed:
                syntax_check = await self._check_syntax(workspace / f if workspace else Path(f))
                if syntax_check:
                    checks.append(syntax_check)

        all_passed = all(c.passed for c in checks)
        return VerificationResult(
            tier=1,
            passed=all_passed,
            checks=checks,
            feedback=self._build_feedback(checks) if not all_passed else None,
        )

    async def _check_syntax(self, path: Path) -> Check | None:
        """Run syntax checks based on file extension."""
        ext = path.suffix
        if ext == ".py":
            proc = await asyncio.create_subprocess_exec(
                "python", "-c", f"import ast; ast.parse(open('{path}').read())",
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await proc.communicate()
            return Check(name=f"syntax_{path.name}", passed=proc.returncode == 0, detail=stderr.decode() if proc.returncode != 0 else None)
        elif ext == ".json":
            try:
                json.loads(path.read_text())
                return Check(name=f"syntax_{path.name}", passed=True)
            except json.JSONDecodeError as e:
                return Check(name=f"syntax_{path.name}", passed=False, detail=str(e))
        elif ext in (".yaml", ".yml"):
            try:
                yaml.safe_load(path.read_text())
                return Check(name=f"syntax_{path.name}", passed=True)
            except yaml.YAMLError as e:
                return Check(name=f"syntax_{path.name}", passed=False, detail=str(e))
        return None
```

## Tier 2: LLM Verification

```python
class LLMVerifier:
    """
    Independent LLM assessment using a DIFFERENT model instance
    with NO prior context from the execution.
    """

    async def verify(self, subtask, result, tool_calls, workspace) -> VerificationResult:
        # Use verifier role — deliberately different from executor
        model = self._model_router.select(tier=1, role="verifier")

        prompt = f"""You are a verification assistant. You are reviewing the output of a completed subtask.

SUBTASK DESCRIPTION:
{subtask.description}

SUBTASK OUTPUT:
{result}

TOOL CALLS MADE:
{self._format_tool_calls(tool_calls)}

ACCEPTANCE CRITERIA:
{subtask.acceptance_criteria or "The subtask description should be fully satisfied."}

Evaluate whether the output correctly satisfies the subtask description and acceptance criteria.

Respond with ONLY a JSON object:
{{
  "passed": true/false,
  "issues": ["list of specific problems found"],
  "confidence": 0.0-1.0,
  "suggestion": "what should be fixed if failed"
}}"""

        response = await model.complete([{"role": "user", "content": prompt}])
        assessment = json.loads(response.text)

        return VerificationResult(
            tier=2,
            passed=assessment["passed"],
            confidence=assessment.get("confidence", 0.5),
            checks=[Check(name="llm_assessment", passed=assessment["passed"],
                         detail="; ".join(assessment.get("issues", [])))],
            feedback=assessment.get("suggestion"),
        )
```

## Tier 3: Voting Verification

```python
class VotingVerifier:
    """
    Run the subtask N times independently, compare outputs.
    Majority agreement = pass. Divergence = human review.
    """

    async def verify(self, subtask, result, tool_calls, workspace) -> VerificationResult:
        vote_count = self._config.tier3_vote_count  # default: 3

        # Run N independent Tier 2 verifications
        tasks = [
            self._tier2.verify(subtask, result, tool_calls, workspace)
            for _ in range(vote_count)
        ]
        results = await asyncio.gather(*tasks)

        pass_count = sum(1 for r in results if r.passed)
        majority = pass_count > vote_count / 2

        return VerificationResult(
            tier=3,
            passed=majority,
            confidence=pass_count / vote_count,
            checks=[Check(
                name="voting",
                passed=majority,
                detail=f"{pass_count}/{vote_count} verifiers agreed output is correct",
            )],
            feedback="Divergent verification — flagged for human review" if not majority else None,
        )
```

## Data Structures

```python
@dataclass
class Check:
    name: str
    passed: bool
    detail: str | None = None

@dataclass
class VerificationResult:
    tier: int                           # Which tier produced this result
    passed: bool
    confidence: float = 1.0             # 0.0-1.0, used for human-in-the-loop gating
    checks: list[Check] = field(default_factory=list)
    feedback: str | None = None         # Actionable feedback for retry
```

## Integration with Retry Logic

When verification fails, the feedback is fed back to the executor for retry (Spec 13):

```python
# In orchestrator, after verification failure:
if not verification.passed and subtask.retry_count < subtask.max_retries:
    subtask.retry_count += 1
    # Feed verification feedback into next attempt
    retry_context = f"Previous attempt failed verification: {verification.feedback}"
    # Re-execute with added context
    ...
```

## Acceptance Criteria

- [ ] Tier 1 checks run on every subtask completion (file existence, syntax, tool success)
- [ ] Tier 2 uses a different model/instance than the executor
- [ ] Tier 2 receives no conversation history from execution (fresh context only)
- [ ] Tier 3 runs N independent verifications and computes majority
- [ ] Verification feedback is structured and actionable for retry
- [ ] Python, JSON, and YAML syntax checking works
- [ ] Tier configuration can be set per-subtask and globally
- [ ] Verification results are stored in task state and event log
- [ ] Failed verification with feedback triggers retry (not immediate failure)
