# Loom: Local LLM Agentic Task Orchestrator

> Loom harnesses local LLMs to finish complex tasks. It decomposes work, drives execution through verified steps, and keeps models on track with structured state instead of chat history. Routes between thinking and acting models, verifies outputs independently, and exposes a clean API for both humans and agents.

## Project Overview

Loom is a local-first task orchestration engine that brings Claude Code/Cowork-style agentic capabilities to local LLM models. It solves the core problem of LLMs completing one subtask and stopping by implementing a harness-driven execution loop with structured state management, multi-tier verification, and intelligent model routing.

## Target Hardware

- Mac Studio with 512GB unified memory (Apple Silicon)
- Primary model: MiniMax M2.1 (Q4, ~130GB) — 230B total / 10B active MoE
- Secondary model: Qwen3 8B or 14B (~5-10GB) — utility/validation model
- Inference runtimes: MLX (primary), Ollama (utility)
- Remaining ~350GB+ free for KV cache, system overhead, Docker containers

## Architecture Philosophy

1. **Harness drives work, not model** — The model is a reasoning engine called repeatedly with scoped prompts. The orchestrator loop decides what happens next.
2. **Structured state over conversation** — No chat history rot. Deterministic retrieval from SQLite, not lossy compaction of message logs.
3. **Verification as separate concern** — Never trust the model to check its own work. Independent verification at every tier.
4. **Write-time extraction** — Extract and tag memory entries when context is fresh, not at query time.
5. **Stateless subtasks** — Fresh context per subtask eliminates accumulated noise and context pollution.
6. **Model-agnostic design** — Interface allows swapping models without changing the orchestrator.
7. **API-first** — Both humans and agents are clients of the same REST+SSE service.
8. **Event-driven observability** — Everything emits structured events for monitoring, replay, and learning.

## System Components and Spec Documents

| # | Spec | Component | Priority |
|---|------|-----------|----------|
| 01 | `01-PROJECT-STRUCTURE.md` | Repository layout, dependencies, configuration | P0 |
| 02 | `02-ORCHESTRATOR-LOOP.md` | Core agentic loop, subtask execution, re-planning | P0 |
| 03 | `03-TASK-STATE-MEMORY.md` | Three-layer memory architecture, SQLite schema | P0 |
| 04 | `04-MODEL-ROUTER.md` | Model selection, routing, think/act mode switching | P0 |
| 05 | `05-TOOL-SYSTEM.md` | Tool definitions, execution, file ops, workspace | P0 |
| 06 | `06-VERIFICATION-GATES.md` | Three-tier verification, validation, acceptance | P1 |
| 07 | `07-API-SERVER.md` | FastAPI REST endpoints, SSE streaming, webhooks | P0 |
| 08 | `08-EVENT-SYSTEM.md` | Event bus, structured logging, correlation IDs | P1 |
| 09 | `09-TUI-CLIENT.md` | Terminal UI with textual, task monitoring, controls | P1 |
| 10 | `10-AGENT-INTEGRATION.md` | MCP server, agent-as-client, external agent API | P1 |
| 11 | `11-WORKSPACE-FILES.md` | Working directory mounting, changelog, file ops | P0 |
| 12 | `12-PROMPT-ARCHITECTURE.md` | Prompt templates, context assembly, constraints | P0 |
| 13 | `13-ERROR-RECOVERY.md` | Retry logic, escalation ladder, failure archetypes | P1 |
| 14 | `14-HUMAN-IN-THE-LOOP.md` | Confidence scoring, approval gates, steering | P1 |
| 15 | `15-LEARNING-SYSTEM.md` | Persistent skill database, cold start, pattern DB | P2 |

## Implementation Order

### Phase 1: Core Engine (P0)
Build the engine that can accept a task, decompose it, execute subtasks, and report results.

1. `01-PROJECT-STRUCTURE.md` — Scaffold the repo
2. `03-TASK-STATE-MEMORY.md` — SQLite schema and state management
3. `12-PROMPT-ARCHITECTURE.md` — Prompt templates (needed before loop)
4. `04-MODEL-ROUTER.md` — Connect to local models
5. `05-TOOL-SYSTEM.md` — Register tools the model can call
6. `11-WORKSPACE-FILES.md` — Mount working directories
7. `02-ORCHESTRATOR-LOOP.md` — Wire it all together
8. `07-API-SERVER.md` — Expose via FastAPI

### Phase 2: Reliability Layer (P1)
Add verification, error recovery, and human oversight.

9. `06-VERIFICATION-GATES.md`
10. `13-ERROR-RECOVERY.md`
11. `08-EVENT-SYSTEM.md`
12. `14-HUMAN-IN-THE-LOOP.md`
13. `09-TUI-CLIENT.md`
14. `10-AGENT-INTEGRATION.md`

### Phase 3: Intelligence Layer (P2)
Add learning and adaptation.

15. `15-LEARNING-SYSTEM.md`

## Key Design Decisions

- **Python 3.12+** with asyncio throughout
- **FastAPI** for the API server (native async, SSE support, OpenAPI docs)
- **SQLite** for all persistent storage (state, memory, events, learning)
- **YAML** for task state representation (human-readable in prompts)
- **TOML** for system configuration
- **Docker** for sandboxed code execution (optional, Phase 2)
- **textual** for terminal UI
- **No frameworks** (no LangChain, no CrewAI) — purpose-built for control and debuggability
