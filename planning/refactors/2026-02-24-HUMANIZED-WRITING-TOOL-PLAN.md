# Humanized Writing Tool Plan (2026-02-24)

## Objective
Create a built-in Loom tool that helps draft and revise content (creative copy, long-form drafts, marketing text, reports) so outputs read more like natural human writing while preserving clarity, factuality, and user intent.

Proposed tool name: `humanize_writing`.

## Product Constraints
1. The tool improves writing quality and naturalness; it must not optimize for evading AI detectors.
2. Deterministic analysis should be first-class (stable, testable, cheap).
3. LLM-dependent steps should be optional/advisory, not required to return useful output.
4. Results must be auditable: return scores, rationale, and concrete revision actions.
5. Integrate with existing Loom tool auto-discovery and safety/path constraints.

## Repo Baseline (What Loom Already Has)
1. Built-in tools are auto-discovered via `Tool.__init_subclass__` and `discover_tools()` (`src/loom/tools/__init__.py`).
2. Tool contracts use JSON schema and `ToolResult` payloads (`src/loom/tools/registry.py`).
3. Existing deterministic scoring pattern exists in `peer_review_simulator` (`src/loom/tools/peer_review_simulator.py`).
4. Existing write artifact path exists in `document_write` (`src/loom/tools/document_write.py`).
5. Existing research tool tests can be extended in `tests/test_research_tools.py`.

## Research Synthesis Driving the Design
1. Decoding and repetition matter: neural text often becomes bland/repetitive under poor decoding; diversity-aware strategies improve quality (Holtzman et al., 2020).
2. Human-likeness needs distribution-aware evaluation: MAUVE correlates with human judgments better than many older distributional metrics (Pillutla et al., 2021).
3. Style control is feasible and practical with controllable generation approaches (Keskar et al., 2019; Dathathri et al., 2020; Smith et al., 2020).
4. Iterative self-feedback improves human preference over one-shot generation (Madaan et al., 2023), suggesting a tool-assisted revise-and-rescore loop.
5. Human preference alignment is stronger with human feedback than scale alone (Ouyang et al., 2022), so plan includes human evaluation in rollout.
6. Human eval design quality is critical; rank-based and stability-focused protocols improve reliability (Novikova et al., 2018; Riley et al., 2024).
7. Detector optimization is the wrong target: text-detection methods face OOD/attack limits and fairness concerns (Wu et al., 2025; Yang et al., 2024; Liang et al., 2023; OpenAI updates 2023-07-20 and 2024-08-04).

## v1 Tool Contract

### Operations
1. `analyze`: score a draft and return actionable humanization feedback.
2. `plan_rewrite`: produce prioritized revision instructions by section/paragraph.
3. `evaluate`: score a revised draft against a target style profile and acceptance threshold.
4. `compare`: compare two drafts and summarize whether naturalness improved/regressed.

### Inputs (Core)
1. `content` or `path` (one required).
2. `mode`: `creative_copy | blog_post | email | report | social_post | custom`.
3. `audience`: short audience descriptor.
4. `voice_profile`: optional structured controls (formality, warmth, assertiveness, humor, sentence_rhythm, lexical_complexity).
5. `constraints`: preserve terms, banned phrases, max reading level, max sentence length, etc.
6. `target_score`: optional `0-100` threshold for loop termination.

### Outputs (Core)
1. `humanization_score` (`0-100`) with sub-scores.
2. `metrics` (deterministic feature values + deltas for compare/evaluate).
3. `issues` (ordered, concrete writing problems).
4. `recommended_edits` (high-impact rewrite instructions).
5. `pass` boolean (if `target_score` provided).
6. Optional artifacts: markdown report and json report.

## Proposed Scoring Framework (Deterministic First)
Use a weighted score with interpretable components:
1. Repetition and degeneration:
   - repeated n-gram ratio
   - near-duplicate sentence ratio
   - consecutive sentence opening similarity
2. Rhythm and variance:
   - sentence-length variance (too-flat rhythm penalty)
   - punctuation variety and cadence balance
3. Lexical diversity:
   - type/token proxy
   - repeated high-frequency word concentration
4. Concreteness and specificity proxies:
   - abstract filler phrase density
   - numeric/detail/entity density
5. Readability and clarity:
   - sentence length bounds
   - passive voice estimate
   - modifier bloat estimate (adverb/adjective stacks, hedge density)
6. Discourse coherence cues:
   - transition variety
   - paragraph topic drift heuristics

Output should include both raw feature values and normalized sub-scores so the model can revise intentionally.

## Architecture and File Plan

### New Modules
1. `src/loom/tools/humanize_writing.py`
2. `src/loom/writing/__init__.py`
3. `src/loom/writing/metrics.py`
4. `src/loom/writing/style_profiles.py`
5. `src/loom/writing/rewrite_plan.py`
6. `src/loom/writing/reporting.py`

### Data/Model Extensions
1. Add dataclasses in `src/loom/research/models.py` (or `src/loom/writing/models.py`) for:
   - `WritingMetricSet`
   - `HumanizationIssue`
   - `HumanizationReport`
   - `StyleProfile`

### Optional Config Additions
1. `writing_humanizer_default_target_score` (default 72)
2. `writing_humanizer_max_content_chars` (default 80_000)
3. `writing_humanizer_mode_weights.<mode>`
4. `writing_humanizer_enable_compare_artifacts` (default true)

## Execution Pattern in Loom
1. Model drafts content.
2. Model calls `humanize_writing(analyze)` on the draft.
3. Model rewrites using returned `recommended_edits`.
4. Model calls `humanize_writing(evaluate)` on revised draft.
5. Stop when score crosses threshold or iteration budget reached (recommend max 2-3 loops per subtask).

This mirrors evidence from iterative refinement research while keeping tool behavior deterministic and testable.

## v1 Workstreams

### W1: Contracts and Metrics
1. Define JSON schema for all operations.
2. Implement deterministic feature extraction and weighted scoring.
3. Finalize `mode`-specific score weights.

Exit criteria:
1. Stable score output for fixed fixtures.
2. Clear failure semantics for invalid input/path/workspace.

### W2: Rewrite Planner
1. Build issue prioritization (`impact`, `severity`, `location`).
2. Generate paragraph-level rewrite instructions with preservation constraints.
3. Add `compare` diff summary + score delta computation.

Exit criteria:
1. `plan_rewrite` consistently returns executable, non-generic edits.
2. `compare` identifies regressions (e.g., increased repetition).

### W3: Tool Integration
1. Implement `HumanizeWritingTool(Tool)` in `src/loom/tools/humanize_writing.py`.
2. Validate workspace-safe path reading/writing.
3. Emit markdown/json artifact reports (optional output paths).

Exit criteria:
1. Tool appears in auto-discovered registry without manual registration.
2. Tool outputs include both human-readable and structured machine-usable data.

### W4: Testing and Evaluation Harness
1. Add unit tests for metric functions and score normalization.
2. Add integration tests in `tests/test_research_tools.py` (or `tests/test_new_tools.py`).
3. Add fixture corpus: human-written samples + intentionally robotic samples + mixed-quality drafts.

Exit criteria:
1. Deterministic tests pass with stable expected values.
2. Regression tests catch score drift from future changes.

### W5: Docs and Adoption
1. Update `README.md` built-in tool list and usage examples.
2. Add usage docs in `docs/` with recommended writing loop.
3. Add process guidance snippet for writing-heavy workflows.

Exit criteria:
1. Users can discover and apply the tool without source inspection.

## Human Evaluation Plan (Required for “ensures human-like” claim)
1. Build a prompt set across modes: creative copy, email, report, social, landing page.
2. For each prompt, compare:
   - baseline one-shot draft
   - tool-assisted revised draft
3. Collect pairwise human preferences and criterion scores:
   - naturalness
   - clarity
   - coherence
   - persuasiveness (when relevant)
4. Use RankME-style relative judgments and stability checks across repeated samples.
5. Launch gate for v1:
   - tool-assisted drafts win pairwise preference in >=60% of prompts
   - no significant factuality/coherence regressions
   - stable ranking under repeat evaluation slices

## Safety and Abuse Guardrails
1. Do not expose “AI detector evasion” objective or score.
2. Include explicit output note: the tool optimizes readability, voice consistency, and flow, not authorship obfuscation.
3. Keep provenance-compatible posture:
   - avoid claims that text is “guaranteed human-written”
   - document detector limitations and fairness risks
4. Respect protected-style policy:
   - add optional blocklist for “imitate living individual exact style” requests if product policy requires it.

## Rollout
1. Phase 1: `analyze` + `evaluate` (deterministic core).
2. Phase 2: `plan_rewrite` + `compare`.
3. Phase 3: docs, evaluation report, threshold tuning.
4. Phase 4: optional advanced style-profile import from user-provided writing samples.

## Acceptance Criteria
1. `humanize_writing` is available in `/tools` and auto-discovered.
2. Tool returns stable, interpretable scoring + concrete revision guidance.
3. Writing workflows show measurable human preference lift in controlled eval.
4. No detector-evasion features are introduced.
5. Tests and docs cover the full v1 contract.

## Risks and Mitigations
1. Risk: score overfits heuristics and rewards formulaic writing.
   Mitigation: combine multiple orthogonal features and verify with human eval slices.
2. Risk: style quality gains reduce factual precision.
   Mitigation: run with existing fact-check / verification passes in high-stakes workflows.
3. Risk: mode defaults mismatch domain voice.
   Mitigation: expose `voice_profile` and `mode` presets with explicit overrides.
4. Risk: metric drift over time.
   Mitigation: fixture-based regression tests + frozen benchmark set.

## Primary Sources
1. Holtzman et al., *The Curious Case of Neural Text Degeneration* (ICLR 2020): https://arxiv.org/abs/1904.09751
2. Pillutla et al., *MAUVE* (NeurIPS 2021): https://arxiv.org/abs/2102.01454
3. Keskar et al., *CTRL* (2019): https://arxiv.org/abs/1909.05858
4. Dathathri et al., *Plug and Play Language Models* (ICLR 2020): https://arxiv.org/abs/1912.02164
5. Smith et al., *Controlling Style in Generated Dialogue* (2020): https://arxiv.org/abs/2009.10855
6. Ouyang et al., *Training language models to follow instructions with human feedback* (2022): https://arxiv.org/abs/2203.02155
7. Madaan et al., *Self-Refine* (2023): https://arxiv.org/abs/2303.17651
8. Novikova et al., *RankME* (NAACL 2018): https://aclanthology.org/N18-2012/
9. Riley et al., *Finding Replicable Human Evaluations via Stable Ranking Probability* (NAACL 2024): https://aclanthology.org/2024.naacl-long.275/
10. Wu et al., *A Survey on LLM-Generated Text Detection* (Computational Linguistics 2025): https://aclanthology.org/2025.cl-1.8/
11. Yang et al., *A Survey on Detection of LLMs-Generated Content* (EMNLP Findings 2024): https://aclanthology.org/2024.findings-emnlp.572/
12. Liang et al., *GPT detectors are biased against non-native English writers* (Patterns 2023): https://pubmed.ncbi.nlm.nih.gov/37521038/
13. OpenAI update (July 20, 2023) on classifier accuracy limits: https://openai.com/index/new-ai-classifier-for-indicating-ai-written-text/
14. OpenAI text provenance update (Aug 4, 2024) on watermarking limitations and metadata direction: https://openai.com/index/understanding-the-source-of-what-we-see-and-hear-online/
15. U.S. Digital.gov plain language guidance: https://digital.gov/guides/plain-language
16. U.S. Digital.gov “Short and simple” principles: https://digital.gov/guides/plain-language/principles/short-simple
