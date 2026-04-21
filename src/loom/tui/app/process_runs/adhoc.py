"""Ad hoc process synthesis and caching helpers."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loom import __version__
from loom.engine.verification.development import (
    development_helper_tool_guidance,
    preferred_development_build_tools,
    recommended_development_build_tools,
)

from ..models import AdhocProcessCacheEntry

if TYPE_CHECKING:
    from loom.processes.schema import ProcessDefinition

logger = logging.getLogger(__name__)


def adhoc_cache_key(goal: str) -> str:
    """Build a stable cache key for a run goal."""
    normalized = " ".join(str(goal or "").strip().lower().split())
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()
    return digest[:16]

def adhoc_cache_dir() -> Path:
    """Return on-disk cache directory for ad hoc process specs."""
    return Path.home() / ".loom" / "cache" / "adhoc-processes"

def adhoc_synthesis_log_path(self) -> Path:
    """Return diagnostics log path for ad hoc synthesis internals."""
    configured = getattr(getattr(self._config, "logging", None), "event_log_path", "")
    root = Path(str(configured).strip()).expanduser() if str(configured).strip() else (
        Path.home() / ".loom" / "logs"
    )
    return root / "adhoc-synthesis.jsonl"

def adhoc_synthesis_artifact_root(self) -> Path:
    """Return directory root for per-run ad hoc synthesis artifacts."""
    return self._adhoc_synthesis_log_path().parent / "adhoc-synthesis"

def create_adhoc_synthesis_artifact_dir(self, *, key: str, goal: str) -> Path | None:
    """Create a per-run artifact directory for ad hoc synthesis."""
    try:
        root = self._adhoc_synthesis_artifact_root()
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        goal_slug = self._sanitize_kebab_token(
            goal,
            fallback="goal",
            max_len=24,
        )
        run_id = uuid.uuid4().hex[:6]
        run_dir = root / f"{stamp}-{key}-{goal_slug}-{run_id}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir
    except Exception as e:
        logger.warning("Failed creating ad hoc synthesis artifact dir: %s", e)
        return None

def write_adhoc_synthesis_artifact_text(
    artifact_dir: Path | None,
    filename: str,
    content: str,
) -> None:
    """Write a text artifact into the synthesis run directory."""
    if artifact_dir is None:
        return
    try:
        path = artifact_dir / filename
        path.write_text(str(content or ""), encoding="utf-8")
    except Exception as e:
        logger.warning("Failed writing ad hoc synthesis artifact %s: %s", filename, e)

def write_adhoc_synthesis_artifact_yaml(
    artifact_dir: Path | None,
    filename: str,
    payload: dict[str, Any] | None,
) -> None:
    """Write a YAML artifact into the synthesis run directory."""
    if artifact_dir is None or not isinstance(payload, dict):
        return
    try:
        import yaml

        path = artifact_dir / filename
        path.write_text(
            yaml.safe_dump(
                payload,
                sort_keys=False,
                allow_unicode=True,
            ),
            encoding="utf-8",
        )
    except Exception as e:
        logger.warning("Failed writing ad hoc synthesis YAML artifact %s: %s", filename, e)

def append_adhoc_synthesis_log(self, payload: dict[str, Any]) -> Path | None:
    """Append one ad hoc synthesis diagnostic entry to disk."""
    if not isinstance(payload, dict) or not payload:
        return None
    path = self._adhoc_synthesis_log_path()
    record = {
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        **payload,
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return path
    except Exception as e:
        logger.warning("Failed writing ad hoc synthesis log %s: %s", path, e)
        return None

def adhoc_cache_path(self, key: str) -> Path:
    """Return cache file path for an ad hoc cache key."""
    safe_key = self._sanitize_kebab_token(
        str(key or ""),
        fallback="adhoc",
        max_len=64,
    ).replace("-", "")
    if not safe_key:
        safe_key = "adhoc"
    return self._adhoc_cache_dir() / f"{safe_key}.yaml"

def adhoc_legacy_cache_path(self, key: str) -> Path:
    """Return legacy JSON cache file path for ad hoc cache key."""
    safe_key = self._sanitize_kebab_token(
        str(key or ""),
        fallback="adhoc",
        max_len=64,
    ).replace("-", "")
    if not safe_key:
        safe_key = "adhoc"
    return self._adhoc_cache_dir() / f"{safe_key}.json"

def persist_adhoc_cache_entry(self, entry: AdhocProcessCacheEntry) -> Path:
    """Persist synthesized ad hoc process definition to ~/.loom/cache."""
    import yaml

    cache_path = self._adhoc_cache_path(entry.key)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    spec_payload = entry.spec or self._spec_from_process_defn(
        entry.process_defn,
        recommended_tools=entry.recommended_tools,
    )
    payload: dict[str, Any] = {
        "key": entry.key,
        "goal": entry.goal,
        "generated_at_monotonic": float(entry.generated_at),
        "saved_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "spec": spec_payload,
    }
    cache_path.write_text(
        yaml.safe_dump(
            payload,
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    return cache_path

def load_adhoc_cache_entry_from_disk(self, key: str) -> AdhocProcessCacheEntry | None:
    """Load ad hoc process cache entry from ~/.loom/cache when present."""
    import yaml

    cache_path = self._adhoc_cache_path(key)
    legacy_path = self._adhoc_legacy_cache_path(key)
    read_path: Path | None = None
    if cache_path.exists():
        read_path = cache_path
    elif legacy_path.exists():
        read_path = legacy_path
    else:
        return None

    try:
        raw_text = read_path.read_text(encoding="utf-8")
        if read_path.suffix.lower() == ".json":
            payload = json.loads(raw_text)
        else:
            payload = yaml.safe_load(raw_text)
    except Exception as e:
        logger.warning("Failed to read ad hoc cache '%s': %s", read_path, e)
        return None
    if not isinstance(payload, dict):
        return None

    goal = str(payload.get("goal", "")).strip()
    raw_spec = payload.get("spec")
    if not goal or not isinstance(raw_spec, dict):
        return None

    raw_intent = self._normalize_adhoc_intent(
        str(raw_spec.get("intent", "")),
        default="",
    )
    if not raw_intent:
        # Legacy cache entries (pre-intent) are treated as stale so /run
        # re-synthesizes with LLM-selected intent.
        return None

    normalized = self._normalize_adhoc_spec(
        raw_spec,
        goal=goal,
        key=key,
        available_tools=self._available_tool_names(),
        intent=raw_intent,
    )
    entry = self._build_adhoc_cache_entry(
        key=key,
        goal=goal,
        spec=normalized,
    )
    generated_at = payload.get("generated_at_monotonic")
    if isinstance(generated_at, (int, float)):
        entry.generated_at = float(generated_at)
    if read_path.suffix.lower() == ".json":
        try:
            self._persist_adhoc_cache_entry(entry)
        except Exception:
            pass
    return entry

def sanitize_synthesis_trace(raw: dict[str, Any] | None) -> dict[str, Any]:
    """Preserve only simple scalar fields from synthesis diagnostics."""
    if not isinstance(raw, dict):
        return {}
    clean: dict[str, Any] = {}
    for key, value in raw.items():
        name = str(key or "").strip()
        if not name:
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            clean[name] = value
    return clean


def sanitize_kebab_token(value: str, *, fallback: str, max_len: int = 48) -> str:
    """Normalize free-form text into a safe kebab-case token."""
    token = re.sub(r"[^a-z0-9]+", "-", str(value or "").lower()).strip("-")
    if not token:
        token = fallback
    if len(token) > max_len:
        token = token[:max_len].strip("-")
    if not token:
        token = fallback
    return token


def sanitize_deliverable_name(value: str, *, fallback: str) -> str:
    """Normalize deliverable path names for generated ad hoc processes."""
    raw = str(value or "").strip()
    if not raw:
        return fallback
    raw = raw.split("—")[0].split(" - ")[0].strip()
    raw = raw.replace("\\", "/").lstrip("/")
    parts = [p for p in raw.split("/") if p and p not in {".", ".."}]
    if not parts:
        return fallback
    safe_parts: list[str] = []
    for part in parts:
        safe = re.sub(r"[^A-Za-z0-9._-]+", "-", part).strip("-")
        if safe:
            safe_parts.append(safe)
    if not safe_parts:
        return fallback
    candidate = "/".join(safe_parts)
    if "." not in Path(candidate).name:
        candidate += ".md"
    return candidate


def available_tool_names(self) -> list[str]:
    """Return sorted available tool names from the active registry."""
    try:
        tools = self._tools.list_tools(runnable_only=True)
    except Exception:
        return []
    names = sorted({
        str(name or "").strip()
        for name in tools
        if str(name or "").strip()
    })
    return names


def adhoc_package_contract_hint(self) -> str:
    """Load full package authoring reference doc for ad hoc synthesis."""
    cached = self._adhoc_package_doc_cache
    if isinstance(cached, str) and cached:
        return cached

    candidates = [
        Path(__file__).resolve().parents[3] / "docs" / "creating-packages.md",
        self._workspace / "docs" / "creating-packages.md",
        Path.cwd() / "docs" / "creating-packages.md",
    ]
    for path in candidates:
        try:
            if path.exists() and path.is_file():
                doc = path.read_text(encoding="utf-8")
                text = (
                    f"Reference document path: {path}\n"
                    "Use this full reference when designing the ad hoc process "
                    "package contract.\n\n"
                    f"{doc}"
                )
                self._adhoc_package_doc_cache = text
                return text
        except Exception:
            continue

    fallback = (
        "Reference document unavailable: docs/creating-packages.md.\n"
        "Use Loom package conventions: kebab-case name, schema_version: 2, "
        "phases with deliverables and acceptance criteria, strict|guided|suggestive "
        "phase_mode, and tools.required drawn from available tools."
    )
    self._adhoc_package_doc_cache = fallback
    return fallback

def synthesis_preview(text: str, *, max_chars: int = 480) -> str:
    """Compact a model response preview for diagnostics logs."""
    normalized = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(normalized) <= max_chars:
        return normalized
    return normalized[:max_chars].rstrip() + "..."

def raw_adhoc_spec_needs_minimal_retry(raw: dict[str, Any] | None) -> bool:
    """Return True when parsed raw spec is too incomplete to trust directly."""
    if not isinstance(raw, dict):
        return True
    phases_raw = raw.get("phases", [])
    if not isinstance(phases_raw, list):
        return True
    phase_rows = [item for item in phases_raw if isinstance(item, dict)]
    # Keep this permissive: simple goals can legitimately need ~3 phases.
    if len(phase_rows) < 3:
        return True
    for phase in phase_rows:
        if not str(phase.get("description", "")).strip():
            return True
        raw_deliverables = phase.get("deliverables", [])
        if isinstance(raw_deliverables, str):
            raw_deliverables = [raw_deliverables]
        if not isinstance(raw_deliverables, list):
            return True
        if not any(str(item or "").strip() for item in raw_deliverables):
            return True
    return False

def normalize_adhoc_intent(intent: str, *, default: str = "research") -> str:
    """Normalize ad hoc intent labels to a supported enum-like value."""
    text = str(intent or "").strip().lower()
    if text in {"research", "writing", "build"}:
        return text
    fallback = str(default or "").strip().lower()
    if not fallback:
        return ""
    if fallback in {"research", "writing", "build"}:
        return fallback
    return "research"

def normalize_adhoc_risk_level(risk_level: str, *, default: str = "") -> str:
    """Normalize ad hoc risk level to the process schema enum."""
    from loom.processes.schema import ProcessDefinition

    normalized = ProcessDefinition._normalize_risk_level(risk_level)
    if normalized:
        return normalized
    return ProcessDefinition._normalize_risk_level(default)


def phases_satisfy_intent(phases: list[dict[str, Any]], intent: str) -> bool:
    """Check whether phase set includes intent-critical steps."""
    phase_texts = [
        (
            f"{str(phase.get('id', '')).strip()} "
            f"{str(phase.get('description', '')).strip()}"
        ).lower()
        for phase in phases
        if isinstance(phase, dict)
    ]
    if not phase_texts:
        return False

    def _has_any(markers: tuple[str, ...]) -> bool:
        return any(any(marker in text for marker in markers) for text in phase_texts)

    if intent == "build":
        return _has_any(("implement", "build", "execute", "develop", "code")) and _has_any(
            ("test", "verify", "validate", "qa"),
        )
    if intent == "writing":
        return _has_any(("draft", "write", "compose")) and _has_any(
            ("revise", "edit", "review", "verify"),
        )
    return _has_any(("collect", "evidence", "research", "source")) and _has_any(
        ("analy", "synth", "compare", "evaluate"),
    )


def is_temperature_one_only_error(value: object) -> bool:
    text = str(value or "").lower()
    return "invalid temperature" in text and "only 1 is allowed" in text


def infer_adhoc_intent_from_phases(cls, phases: list[dict[str, Any]]) -> str:
    """Infer intent from phase semantics when explicit intent is unavailable."""
    if cls._phases_satisfy_intent(phases, "build"):
        return "build"
    if cls._phases_satisfy_intent(phases, "writing"):
        return "writing"
    if cls._phases_satisfy_intent(phases, "research"):
        return "research"
    return "research"


def spec_from_process_defn(
    cls,
    process_defn: ProcessDefinition,
    *,
    recommended_tools: list[str],
) -> dict[str, Any]:
    """Serialize a ProcessDefinition into ad hoc spec-shaped payload."""
    from loom.processes.schema import ProcessDefinition

    phases = [
        {
            "id": str(phase.id or "").strip(),
            "description": str(phase.description or "").strip(),
            "depends_on": [
                str(dep).strip()
                for dep in list(phase.depends_on)
                if str(dep).strip()
            ],
            "acceptance_criteria": str(phase.acceptance_criteria or "").strip(),
            "deliverables": [
                str(item).strip()
                for item in list(phase.deliverables)
                if str(item).strip()
            ],
        }
        for phase in list(process_defn.phases)
    ]
    return {
        "intent": cls._infer_adhoc_intent_from_phases(phases),
        "risk_level": ProcessDefinition._normalize_risk_level(
            getattr(process_defn, "risk_level", ""),
        ),
        "name": str(process_defn.name or "").strip(),
        "description": str(process_defn.description or "").strip(),
        "persona": str(process_defn.persona or "").strip(),
        "phase_mode": str(process_defn.phase_mode or "guided").strip(),
        "tool_guidance": str(process_defn.tool_guidance or "").strip(),
        "required_tools": [
            str(item).strip()
            for item in list(getattr(process_defn.tools, "required", []) or [])
            if str(item).strip()
        ],
        "recommended_tools": [
            str(item).strip()
            for item in recommended_tools
            if str(item).strip()
        ],
        "validity_contract": (
            dict(getattr(process_defn, "validity_contract", {}))
            if isinstance(getattr(process_defn, "validity_contract", {}), dict)
            else {}
        ),
        "verification_policy": process_defn.verification_policy_payload(),
        "phases": phases,
    }

def resolve_adhoc_intent(
    cls,
    raw: dict[str, Any] | None,
    *,
    intent_hint: str | None = None,
) -> str:
    """Resolve intent from model-provided fields, then phase semantics."""
    normalized_hint = cls._normalize_adhoc_intent(
        str(intent_hint or ""),
        default="",
    )
    if normalized_hint:
        return normalized_hint
    if not isinstance(raw, dict):
        return "research"
    for key in ("intent", "goal_intent", "request_intent", "goal_type"):
        value = cls._normalize_adhoc_intent(str(raw.get(key, "")), default="")
        if value:
            return value
    raw_phases = raw.get("phases", [])
    if isinstance(raw_phases, list):
        phase_rows = [item for item in raw_phases if isinstance(item, dict)]
        if phase_rows:
            return cls._infer_adhoc_intent_from_phases(phase_rows)
    return "research"


def extract_json_payload(
    raw_text: str,
    *,
    expected_keys: tuple[str, ...] = (),
) -> dict[str, Any] | None:
    """Best-effort structured payload extraction from model output."""
    text = str(raw_text or "").strip()
    if not text:
        return None
    # Normalize typographic quotes that commonly break JSON parsing.
    text = (
        text.replace("“", '"')
        .replace("”", '"')
        .replace("’", "'")
        .replace("‘", "'")
    )

    def _strip_wrapping_fence(value: str) -> str:
        candidate = value.strip()
        if not candidate.startswith("```"):
            return candidate
        lines = candidate.splitlines()
        if lines:
            lines = lines[1:]
        while lines and not lines[-1].strip():
            lines.pop()
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()

    def _parse_blob(blob: str) -> dict[str, Any] | None:
        source = _strip_wrapping_fence(blob)
        if not source:
            return None

        decoder = json.JSONDecoder()
        candidates: list[dict[str, Any]] = []
        try:
            parsed = decoder.decode(source)
            if isinstance(parsed, dict):
                candidates.append(parsed)
        except Exception:
            pass

        for idx, ch in enumerate(source):
            if ch != "{":
                continue
            try:
                parsed, _end = decoder.raw_decode(source[idx:])
            except Exception:
                continue
            if isinstance(parsed, dict):
                candidates.append(parsed)

        if expected_keys:
            for payload in candidates:
                if all(key in payload for key in expected_keys):
                    return payload
        if candidates:
            return candidates[0]

        # Some models drift into YAML-like output despite JSON instructions.
        try:
            import yaml

            parsed_yaml = yaml.safe_load(source)
            if isinstance(parsed_yaml, dict):
                return parsed_yaml
        except Exception:
            pass
        return None

    payloads: list[dict[str, Any]] = []

    direct = _parse_blob(text)
    if direct is not None:
        payloads.append(direct)

    # Try fenced code blocks anywhere in the response, not just full-body fences.
    for match in re.finditer(r"```(?:json|yaml|yml)?\s*([\s\S]*?)```", text, re.IGNORECASE):
        block = str(match.group(1) or "").strip()
        if not block:
            continue
        parsed = _parse_blob(block)
        if parsed is not None:
            payloads.append(parsed)

    # Try to recover YAML objects from markdown/prose preambles.
    if expected_keys:
        lines = text.splitlines()
        lowered_key_prefixes = tuple(
            f"{str(key).strip().lower()}:"
            for key in expected_keys
            if str(key).strip()
        )
        for idx, line in enumerate(lines):
            lower = str(line).strip().lower()
            if not lowered_key_prefixes:
                break
            if not any(lower.startswith(prefix) for prefix in lowered_key_prefixes):
                continue
            snippet = "\n".join(lines[idx:]).strip()
            if not snippet:
                continue
            parsed = _parse_blob(snippet)
            if parsed is not None:
                payloads.append(parsed)
                break

    if expected_keys:
        for payload in payloads:
            if all(key in payload for key in expected_keys):
                return payload
        # When an explicit schema is expected, avoid returning nested/partial
        # dicts (e.g., truncated JSON where only one phase object parses).
        return None

    return payloads[0] if payloads else None

def resolve_adhoc_risk_level(
    cls,
    *,
    goal: str,
    intent: str,
    raw: dict[str, Any] | None,
) -> str:
    """Resolve risk level from explicit model field, then goal heuristics."""
    if isinstance(raw, dict):
        for key in ("risk_level", "risk", "domain_risk"):
            candidate = cls._normalize_adhoc_risk_level(str(raw.get(key, "")))
            if candidate:
                return candidate

    goal_text = str(goal or "").strip().lower()
    high_risk_tokens = (
        "investment",
        "investing",
        "finance",
        "financial",
        "medical",
        "medicine",
        "healthcare",
        "legal",
        "law",
        "compliance",
    )
    if any(token in goal_text for token in high_risk_tokens):
        return "high"
    if intent in {"research", "writing"}:
        return "medium"
    return "low"

def adhoc_intent_progression(intent: str) -> str:
    """Return phase progression guidance for the inferred intent."""
    if intent == "build":
        return (
            "scope -> implementation plan/design -> implement/build -> "
            "test/verify -> package/handoff -> final delivery"
        )
    if intent == "writing":
        return (
            "scope -> outline -> draft -> revise/edit -> "
            "editorial verification -> final delivery"
        )
    return (
        "scope -> source planning -> evidence collection -> "
        "analysis/synthesis -> verification -> final delivery"
    )

def adhoc_default_validity_contract(intent: str, risk_level: str) -> dict[str, Any]:
    """Return intent-aware validity defaults for ad hoc process synthesis."""
    normalized_intent = str(intent or "").strip().lower()
    normalized_risk = normalize_adhoc_risk_level(risk_level)
    high_risk = normalized_risk in {"high", "critical"}
    require_fact_checker = bool(
        high_risk or normalized_intent in {"research", "writing"},
    )
    min_supported_ratio = 0.8 if normalized_intent in {"research", "writing"} else 0.65
    max_unverified_ratio = 0.2 if normalized_intent in {"research", "writing"} else 0.35
    if high_risk:
        min_supported_ratio = max(min_supported_ratio, 0.8)
        max_unverified_ratio = min(max_unverified_ratio, 0.2)
    temporal_consistency = {
        "enabled": bool(high_risk),
        "require_as_of_alignment": bool(high_risk),
        "enforce_cross_claim_date_conflict_check": bool(high_risk),
        "max_source_age_days": 365 if high_risk else 0,
        "as_of": "",
    }
    contract = {
        "enabled": True,
        "claim_extraction": {"enabled": True},
        "critical_claim_types": ["numeric", "date", "entity_fact"],
        "min_supported_ratio": min_supported_ratio,
        "max_unverified_ratio": max_unverified_ratio,
        "max_contradicted_count": 0,
        "prune_mode": "rewrite_uncertainty",
        "require_fact_checker_for_synthesis": require_fact_checker,
        "final_gate": {
            "enforce_verified_context_only": True,
            "synthesis_min_verification_tier": 2,
            "critical_claim_support_ratio": 1.0,
            "temporal_consistency": temporal_consistency,
        },
    }
    if normalized_intent == "build":
        contract["critical_claim_types"] = ["numeric", "date"]
        contract["prune_mode"] = "drop"
    return contract

def _adhoc_build_verification_policy() -> dict[str, Any]:
    """Return development-oriented verification defaults for build workflows."""
    return {
        "mode": "static_first",
        "static_checks": {
            # Keep true integrity failures hard, fail real build/test checks,
            # and downgrade verifier harness/browser issues that are not
            # evidence of a broken deliverable.
            "tool_success_policy": "development_balanced",
            "phase_scope": "current_phase",
        },
        "semantic_checks": [
            {
                "name": "behavior_over_style_runtime_checks",
                "description": (
                    "Prefer runtime and artifact behavior checks over "
                    "style-only source heuristics for build workflows."
                ),
            },
            {
                "name": "optional_browser_verification",
                "kind": "runtime_probe",
                "capability": "browser_runtime",
                "helper": "browser_assert",
                "target": "ui_surface",
                "optional": True,
            },
            {
                "name": "optional_service_runtime_probe",
                "kind": "service_probe",
                "capability": "service_runtime",
                "helper": "serve_static",
                "target": "local_service",
                "optional": True,
            },
        ],
        "output_contract": {
            "required_fields": ["passed", "outcome", "reason_code", "severity_class"],
            "metadata_fields": [
                "verification_profile",
                "verification_profile_confidence",
                "dev_verification_summary",
            ],
        },
        "outcome_policy": {
            "treat_verifier_infra_as_warning": True,
            "prefer_behavior_over_style": True,
            "optional_capabilities": ["browser_runtime", "service_runtime"],
        },
    }


def adhoc_default_verification_policy(intent: str = "research") -> dict[str, Any]:
    """Return verification policy defaults for synthesized ad hoc processes."""
    normalized_intent = normalize_adhoc_intent(intent, default="research")
    if normalized_intent == "build":
        return _adhoc_build_verification_policy()
    return {
        "mode": "llm_first",
        "static_checks": {
            # Keep safety/integrity failures hard while treating ordinary
            # tool/runtime faults as recoverable method failures to retry.
            "tool_success_policy": "method_resilient",
        },
        "semantic_checks": [],
        "output_contract": {},
        "outcome_policy": {},
    }

def merge_adhoc_verification_policy(
    base: dict[str, Any],
    incoming: object,
) -> dict[str, Any]:
    """Merge model-provided verification policy into ad hoc defaults."""
    merged = {
        "mode": str(base.get("mode", "llm_first") or "llm_first").strip().lower(),
        "static_checks": (
            dict(base.get("static_checks", {}))
            if isinstance(base.get("static_checks", {}), dict)
            else {}
        ),
        "semantic_checks": (
            [item for item in base.get("semantic_checks", []) if isinstance(item, dict)]
            if isinstance(base.get("semantic_checks", []), list)
            else []
        ),
        "output_contract": (
            dict(base.get("output_contract", {}))
            if isinstance(base.get("output_contract", {}), dict)
            else {}
        ),
        "outcome_policy": (
            dict(base.get("outcome_policy", {}))
            if isinstance(base.get("outcome_policy", {}), dict)
            else {}
        ),
    }
    if not isinstance(incoming, dict):
        return merged

    mode = str(incoming.get("mode", "") or "").strip().lower()
    if mode in {"llm_first", "static_first"}:
        merged["mode"] = mode

    static_checks = incoming.get("static_checks")
    if isinstance(static_checks, dict):
        for key, value in static_checks.items():
            merged["static_checks"][str(key)] = value

    semantic_checks = incoming.get("semantic_checks")
    if isinstance(semantic_checks, list):
        merged["semantic_checks"] = [
            item for item in semantic_checks if isinstance(item, dict)
        ]

    output_contract = incoming.get("output_contract")
    if isinstance(output_contract, dict):
        merged["output_contract"].update(output_contract)

    outcome_policy = incoming.get("outcome_policy")
    if isinstance(outcome_policy, dict):
        merged["outcome_policy"].update(outcome_policy)

    return merged

def adhoc_intent_phase_blueprint(intent: str, slug: str) -> list[dict[str, Any]]:
    """Return deterministic phase blueprint for a goal intent."""
    if intent == "build":
        return [
            {
                "id": "scope-and-constraints",
                "description": (
                    "Clarify functional goals, constraints, and acceptance criteria."
                ),
                "depends_on": [],
                "acceptance_criteria": (
                    "Requirements and success criteria are explicit and testable."
                ),
                "deliverables": [f"{slug}-requirements.md"],
            },
            {
                "id": "implementation-plan",
                "description": "Design implementation approach and execution plan.",
                "depends_on": ["scope-and-constraints"],
                "acceptance_criteria": (
                    "Plan maps required changes, affected files, and sequencing."
                ),
                "deliverables": [f"{slug}-implementation-plan.md"],
            },
            {
                "id": "implement-solution",
                "description": "Execute implementation changes to satisfy requirements.",
                "depends_on": ["implementation-plan"],
                "acceptance_criteria": (
                    "Requested solution is implemented and integrated without regressions."
                ),
                "deliverables": [f"{slug}-implementation-summary.md"],
            },
            {
                "id": "test-and-verify",
                "description": "Run validations and verify behavior against requirements.",
                "depends_on": ["implement-solution"],
                "acceptance_criteria": (
                    "Verification evidence confirms behavior, edge cases, and quality."
                ),
                "deliverables": [f"{slug}-verification.md"],
            },
            {
                "id": "package-and-handoff",
                "description": "Prepare final artifacts, notes, and operational guidance.",
                "depends_on": ["test-and-verify"],
                "acceptance_criteria": (
                    "Artifacts and notes are complete, actionable, and ready for handoff."
                ),
                "deliverables": [f"{slug}-handoff.md"],
            },
            {
                "id": "deliver-final",
                "description": "Deliver final outcome with summary of what changed and why.",
                "depends_on": ["package-and-handoff"],
                "acceptance_criteria": (
                    "Final output is complete, validated, and aligned with the goal."
                ),
                "deliverables": [f"{slug}-final.md"],
            },
        ]
    if intent == "writing":
        return [
            {
                "id": "scope-and-constraints",
                "description": (
                    "Clarify audience, objective, constraints, tone, and output format."
                ),
                "depends_on": [],
                "acceptance_criteria": (
                    "Writing brief defines objective, audience, and quality bar."
                ),
                "deliverables": [f"{slug}-brief.md"],
            },
            {
                "id": "outline-and-sources",
                "description": (
                    "Create outline and gather source material or supporting points."
                ),
                "depends_on": ["scope-and-constraints"],
                "acceptance_criteria": (
                    "Outline and supporting material are sufficient for a full draft."
                ),
                "deliverables": [f"{slug}-outline.md"],
            },
            {
                "id": "draft-content",
                "description": "Write the first complete draft.",
                "depends_on": ["outline-and-sources"],
                "acceptance_criteria": (
                    "Draft covers required content and major claims end-to-end."
                ),
                "deliverables": [f"{slug}-draft.md"],
            },
            {
                "id": "revise-and-edit",
                "description": "Revise structure, clarity, and argument quality.",
                "depends_on": ["draft-content"],
                "acceptance_criteria": (
                    "Revisions improve coherence, clarity, and reader usefulness."
                ),
                "deliverables": [f"{slug}-revised.md"],
            },
            {
                "id": "verify-quality",
                "description": "Perform editorial and factual quality checks.",
                "depends_on": ["revise-and-edit"],
                "acceptance_criteria": (
                    "Claims, consistency, and style pass defined quality checks."
                ),
                "deliverables": [f"{slug}-verification.md"],
            },
            {
                "id": "deliver-final",
                "description": "Deliver final polished piece and rationale for key choices.",
                "depends_on": ["verify-quality"],
                "acceptance_criteria": (
                    "Final output is publication-ready for the stated objective."
                ),
                "deliverables": [f"{slug}-final.md"],
            },
        ]
    return [
        {
            "id": "scope-and-constraints",
            "description": (
                "Clarify objective, constraints, assumptions, and success criteria."
            ),
            "depends_on": [],
            "acceptance_criteria": (
                "Scope, constraints, and success criteria are explicit and actionable."
            ),
            "deliverables": [f"{slug}-brief.md"],
        },
        {
            "id": "source-plan",
            "description": (
                "Define research strategy, source classes, and evidence standards."
            ),
            "depends_on": ["scope-and-constraints"],
            "acceptance_criteria": (
                "Research plan lists source priorities, collection method, and "
                "quality checks."
            ),
            "deliverables": [f"{slug}-source-plan.md"],
        },
        {
            "id": "collect-evidence",
            "description": "Gather and verify relevant evidence and source data.",
            "depends_on": ["source-plan"],
            "acceptance_criteria": (
                "Evidence log captures sufficient, credible, and attributable sources."
            ),
            "deliverables": [f"{slug}-evidence.md"],
        },
        {
            "id": "analyze-findings",
            "description": (
                "Analyze evidence, compare alternatives, and synthesize conclusions."
            ),
            "depends_on": ["collect-evidence"],
            "acceptance_criteria": (
                "Analysis explains tradeoffs, uncertainty, and rationale for conclusions."
            ),
            "deliverables": [f"{slug}-analysis.md"],
        },
        {
            "id": "verify-quality",
            "description": "Validate completeness, consistency, and evidentiary support.",
            "depends_on": ["analyze-findings"],
            "acceptance_criteria": (
                "Claims are checked against evidence and key gaps/risks are documented."
            ),
            "deliverables": [f"{slug}-verification.md"],
        },
        {
            "id": "deliver-report",
            "description": "Produce final deliverable with recommendations and citations.",
            "depends_on": ["verify-quality"],
            "acceptance_criteria": (
                "Final output meets goal, includes references, and is ready to share."
            ),
            "deliverables": [f"{slug}-report.md"],
        },
    ]

def fallback_adhoc_spec(
    self,
    goal: str,
    *,
    available_tools: list[str],
    intent: str | None = None,
) -> dict[str, Any]:
    """Return deterministic fallback spec when model synthesis fails."""
    resolved_intent = self._normalize_adhoc_intent(
        str(intent or ""),
        default="research",
    )
    slug = self._sanitize_kebab_token(goal, fallback="adhoc-process", max_len=26)
    available = set(available_tools)
    preferred_by_intent: dict[str, list[str]] = {
        "build": preferred_development_build_tools(available_tools),
        "writing": [
            "read_file",
            "write_file",
            "document_write",
            "search_files",
            "web_search",
        ],
        "research": [
            "search_files",
            "read_file",
            "write_file",
            "document_write",
            "web_search",
            "web_fetch",
            "spreadsheet",
        ],
    }
    preferred = preferred_by_intent.get(
        resolved_intent,
        preferred_by_intent["research"],
    )
    required_tools = [name for name in preferred if name in available][:5]
    if not required_tools and available_tools:
        required_tools = available_tools[: min(5, len(available_tools))]
    recommended_by_intent: dict[str, list[str]] = {
        "build": recommended_development_build_tools(available_tools),
        "writing": ["web_search", "document_write"],
        "research": ["web_search", "web_fetch", "spreadsheet", "calculator"],
    }
    recommended = [
        name
        for name in recommended_by_intent.get(resolved_intent, [])
        if name not in available
    ]
    resolved_risk_level = self._resolve_adhoc_risk_level(
        goal=goal,
        intent=resolved_intent,
        raw=None,
    )
    tool_guidance = (
        "Use available tools aggressively for evidence gathering, verification, "
        "and artifact production. Prefer primary sources, maintain traceability, "
        "and keep outputs concise and decision-oriented."
    )
    if resolved_intent in {"research", "writing"}:
        tool_guidance = (
            f"{tool_guidance}\n\n"
            "Do not treat web_search snippets as final evidence. When search "
            "finds a relevant result, follow up with web_fetch or web_fetch_html "
            "before citing facts. If an exact metric cannot be confirmed, record "
            "the gap explicitly instead of estimating. Run fact_checker on "
            "material factual or numeric conclusions before final synthesis."
        )
    if resolved_intent == "build":
        tool_guidance = (
            f"{tool_guidance}\n\n{development_helper_tool_guidance()}"
        )
    return {
        "source": "fallback_template",
        "intent": resolved_intent,
        "risk_level": resolved_risk_level,
        "name": f"{slug}-adhoc",
        "description": f"Ad hoc process synthesized for goal: {goal.strip()}",
        "persona": (
            "You are a pragmatic analyst. Build a concrete plan, produce useful "
            "artifacts, and state evidence and assumptions."
        ),
        # Guided keeps the planner in control while still providing structure.
        "phase_mode": "guided",
        "tool_guidance": tool_guidance,
        "required_tools": required_tools,
        "recommended_tools": recommended,
        "validity_contract": self._adhoc_default_validity_contract(
            resolved_intent,
            resolved_risk_level,
        ),
        "verification_policy": self._adhoc_default_verification_policy(
            resolved_intent,
        ),
        "phases": self._adhoc_intent_phase_blueprint(resolved_intent, slug),
    }

def normalize_adhoc_spec(
    self,
    raw: dict[str, Any] | None,
    *,
    goal: str,
    key: str,
    available_tools: list[str],
    intent: str | None = None,
) -> dict[str, Any]:
    """Normalize model-produced ad hoc process spec into safe structure."""
    resolved_intent = self._resolve_adhoc_intent(raw, intent_hint=intent)
    resolved_risk_level = self._resolve_adhoc_risk_level(
        goal=goal,
        intent=resolved_intent,
        raw=raw,
    )
    fallback = self._fallback_adhoc_spec(
        goal,
        available_tools=available_tools,
        intent=resolved_intent,
    )
    fallback["risk_level"] = resolved_risk_level
    from loom.processes.schema import ProcessDefinition

    fallback_validity = self._adhoc_default_validity_contract(
        resolved_intent,
        resolved_risk_level,
    )
    fallback["validity_contract"] = fallback_validity
    fallback_verification_policy = self._adhoc_default_verification_policy(
        resolved_intent,
    )
    fallback["verification_policy"] = fallback_verification_policy
    baseline_validity_contract = ProcessDefinition._normalize_validity_contract(
        fallback.get("validity_contract", {}),
    )
    baseline_verification_policy = self._merge_adhoc_verification_policy(
        fallback_verification_policy,
        {},
    )
    raw_validity_contract: object = {}
    if isinstance(raw, dict):
        raw_validity_contract = raw.get("validity_contract", {})
    if isinstance(raw_validity_contract, bool):
        raw_validity_contract = {"enabled": raw_validity_contract}
    if isinstance(raw_validity_contract, dict) and raw_validity_contract:
        validity_contract = ProcessDefinition._merge_validity_contract(
            baseline_validity_contract,
            raw_validity_contract,
        )
    else:
        validity_contract = baseline_validity_contract
    raw_verification_policy: object = {}
    if isinstance(raw, dict):
        raw_verification_policy = raw.get("verification_policy", {})
    verification_policy = self._merge_adhoc_verification_policy(
        baseline_verification_policy,
        raw_verification_policy,
    )
    if not isinstance(raw, dict):
        fallback["validity_contract"] = validity_contract
        fallback["verification_policy"] = verification_policy
        return fallback

    raw_synthesis = self._sanitize_synthesis_trace(
        raw.get("_synthesis") if isinstance(raw.get("_synthesis"), dict) else None
    )
    available_set = set(available_tools)
    proposed_name = (
        str(raw.get("name") or raw.get("name_hint") or "")
        .strip()
        .lower()
    )
    name = self._sanitize_kebab_token(
        proposed_name,
        fallback=f"adhoc-{key[:8]}",
        max_len=40,
    )
    if not name.endswith("-adhoc"):
        name = f"{name}-adhoc"

    description = str(raw.get("description", "")).strip() or fallback["description"]
    persona = str(raw.get("persona", "")).strip() or fallback["persona"]
    tool_guidance = str(raw.get("tool_guidance", "")).strip() or fallback["tool_guidance"]
    if resolved_intent in {"research", "writing"}:
        tool_guidance = (
            f"{tool_guidance}\n\n"
            "Research evidence rule: do not rely on web_search snippets alone for "
            "factual claims. Fetch the underlying page before citing it, mark "
            "unconfirmed metrics as gaps instead of estimates, and use fact_checker "
            "before final synthesis of material claims."
        ).strip()
    valid_phase_modes = {"strict", "guided", "suggestive"}
    raw_phase_mode = str(raw.get("phase_mode", "")).strip().lower()
    fallback_phase_mode = str(fallback.get("phase_mode", "guided")).strip().lower()
    if fallback_phase_mode not in valid_phase_modes:
        fallback_phase_mode = "guided"
    phase_mode = raw_phase_mode if raw_phase_mode in valid_phase_modes else fallback_phase_mode

    required_tools: list[str] = []
    for item in raw.get("required_tools", []):
        tool_name = str(item or "").strip()
        if not tool_name or tool_name not in available_set:
            continue
        if tool_name not in required_tools:
            required_tools.append(tool_name)
    if not required_tools:
        required_tools = list(fallback["required_tools"])

    recommended_tools: list[str] = []
    for item in raw.get("recommended_tools", []):
        tool_name = str(item or "").strip()
        if not tool_name or tool_name in available_set:
            continue
        if tool_name not in recommended_tools:
            recommended_tools.append(tool_name)
    for item in fallback["recommended_tools"]:
        if item not in recommended_tools and item not in available_set:
            recommended_tools.append(item)

    seen_phase_ids: set[str] = set()
    phases: list[dict[str, Any]] = []
    raw_phases = raw.get("phases", [])
    if not isinstance(raw_phases, list):
        raw_phases = []
    for idx, phase in enumerate(raw_phases, start=1):
        if not isinstance(phase, dict):
            continue
        phase_id = self._sanitize_kebab_token(
            str(phase.get("id", "")),
            fallback=f"phase-{idx}",
            max_len=36,
        )
        if phase_id in seen_phase_ids:
            phase_id = f"{phase_id}-{idx}"
        seen_phase_ids.add(phase_id)
        desc = str(phase.get("description", "")).strip()
        if not desc:
            desc = f"Execute {phase_id.replace('-', ' ')}."

        deliverables: list[str] = []
        raw_deliverables = phase.get("deliverables", [])
        if isinstance(raw_deliverables, str):
            raw_deliverables = [raw_deliverables]
        if isinstance(raw_deliverables, list):
            for didx, item in enumerate(raw_deliverables, start=1):
                clean = self._sanitize_deliverable_name(
                    str(item or ""),
                    fallback=f"{phase_id}-{didx}.md",
                )
                if clean not in deliverables:
                    deliverables.append(clean)
        if not deliverables:
            deliverables = [f"{phase_id}.md"]

        depends_on: list[str] = []
        raw_depends = phase.get("depends_on", [])
        if isinstance(raw_depends, str):
            raw_depends = [raw_depends]
        if isinstance(raw_depends, list):
            for dep in raw_depends:
                dep_id = self._sanitize_kebab_token(
                    str(dep or ""),
                    fallback="",
                    max_len=36,
                )
                if not dep_id or dep_id == phase_id:
                    continue
                if dep_id in seen_phase_ids and dep_id not in depends_on:
                    depends_on.append(dep_id)

        phases.append({
            "id": phase_id,
            "description": desc,
            "depends_on": depends_on,
            "deliverables": deliverables,
            "acceptance_criteria": str(
                phase.get("acceptance_criteria", ""),
            ).strip(),
        })

    used_template_phases = False
    enforce_intent_shape = resolved_intent in {"build", "writing"}
    if (
        not phases
        or len(phases) < 3
        or (
            enforce_intent_shape
            and not self._phases_satisfy_intent(phases, resolved_intent)
        )
    ):
        phases = list(fallback["phases"])
        used_template_phases = True

    if recommended_tools:
        recommended = ", ".join(recommended_tools)
        tool_guidance = (
            f"{tool_guidance}\n\nRecommended additional tools for better outcomes: "
            f"{recommended}"
        )

    source = "fallback_template" if used_template_phases else "model_generated"
    raw_source = str(raw.get("source", "")).strip().lower()
    if raw_source in {"fallback_template", "model_generated"}:
        source = raw_source
    elif not used_template_phases:
        # Preserve template provenance when a cache entry omits source but
        # still exactly matches the fallback phase blueprint.
        if self._is_template_like_adhoc_spec(
            {"intent": resolved_intent, "phases": phases},
            goal=goal,
        ):
            source = "fallback_template"

    return {
        "source": source,
        "intent": resolved_intent,
        "risk_level": resolved_risk_level,
        "name": name,
        "description": description,
        "persona": persona,
        "phase_mode": phase_mode,
        "tool_guidance": tool_guidance,
        "required_tools": required_tools,
        "recommended_tools": recommended_tools,
        "validity_contract": validity_contract,
        "verification_policy": verification_policy,
        "phases": phases,
        "_synthesis": raw_synthesis,
    }

def build_adhoc_cache_entry(
    self,
    *,
    key: str,
    goal: str,
    spec: dict[str, Any],
) -> AdhocProcessCacheEntry:
    """Build a cached ad hoc process entry from normalized spec."""
    from loom.processes.schema import (
        PhaseTemplate,
        ProcessDefinition,
        PromptContracts,
        ToolRequirements,
        VerificationPolicyContract,
    )

    phases = [
        PhaseTemplate(
            id=str(phase.get("id", "")).strip(),
            description=str(phase.get("description", "")).strip(),
            depends_on=[
                str(dep).strip()
                for dep in phase.get("depends_on", [])
                if str(dep).strip()
            ],
            acceptance_criteria=str(
                phase.get("acceptance_criteria", ""),
            ).strip(),
            deliverables=[
                str(item).strip()
                for item in phase.get("deliverables", [])
                if str(item).strip()
            ],
        )
        for phase in spec.get("phases", [])
        if isinstance(phase, dict)
    ]
    verification_policy_raw = self._merge_adhoc_verification_policy(
        self._adhoc_default_verification_policy(
            str(spec.get("intent", "")) or "research",
        ),
        spec.get("verification_policy", {}),
    )
    resolved_intent = self._normalize_adhoc_intent(
        str(spec.get("intent", "")),
        default="research",
    )
    process_defn = ProcessDefinition(
        name=str(spec.get("name", "")).strip() or f"adhoc-{key[:8]}",
        version="adhoc-1",
        description=str(spec.get("description", "")).strip(),
        persona=str(spec.get("persona", "")).strip(),
        risk_level=self._normalize_adhoc_risk_level(str(spec.get("risk_level", ""))),
        tool_guidance=str(spec.get("tool_guidance", "")).strip(),
        tools=ToolRequirements(
            required=[
                str(item).strip()
                for item in spec.get("required_tools", [])
                if str(item).strip()
            ],
        ),
        phase_mode=str(spec.get("phase_mode", "guided")).strip() or "guided",
        phases=phases,
        validity_contract=(
            dict(spec.get("validity_contract", {}))
            if isinstance(spec.get("validity_contract", {}), dict)
            else {}
        ),
        prompt_contracts=PromptContracts(
            evidence_contract=(
                {"enabled": True, "applies_to_phases": ["*"]}
                if resolved_intent in {"research", "writing"}
                else {}
            ),
        ),
        verification_policy=VerificationPolicyContract(
            mode=(
                str(verification_policy_raw.get("mode", "llm_first") or "llm_first")
                .strip()
                .lower()
                or "llm_first"
            ),
            static_checks=(
                dict(verification_policy_raw.get("static_checks", {}))
                if isinstance(
                    verification_policy_raw.get("static_checks", {}),
                    dict,
                )
                else {}
            ),
            semantic_checks=(
                [
                    item
                    for item in (
                        verification_policy_raw.get("semantic_checks", [])
                        if isinstance(verification_policy_raw.get("semantic_checks", []), list)
                        else []
                    )
                    if isinstance(item, dict)
                ]
            ),
            output_contract=(
                dict(verification_policy_raw.get("output_contract", {}))
                if isinstance(
                    verification_policy_raw.get("output_contract", {}),
                    dict,
                )
                else {}
            ),
            outcome_policy=(
                dict(verification_policy_raw.get("outcome_policy", {}))
                if isinstance(
                    verification_policy_raw.get("outcome_policy", {}),
                    dict,
                )
                else {}
            ),
        ),
        tags=list(dict.fromkeys(["adhoc", "generated", resolved_intent])),
    )
    try:
        spec_snapshot = json.loads(json.dumps(spec, ensure_ascii=False))
    except Exception:
        spec_snapshot = self._spec_from_process_defn(
            process_defn,
            recommended_tools=[
                str(item).strip()
                for item in spec.get("recommended_tools", [])
                if str(item).strip()
            ],
        )
    return AdhocProcessCacheEntry(
        key=key,
        goal=goal,
        process_defn=process_defn,
        recommended_tools=[
            str(item).strip()
            for item in spec.get("recommended_tools", [])
            if str(item).strip()
        ],
        spec=spec_snapshot if isinstance(spec_snapshot, dict) else {},
    )

def is_template_like_adhoc_spec(self, spec: dict[str, Any], *, goal: str) -> bool:
    """Return True when a spec appears to be the fallback template."""
    if not isinstance(spec, dict):
        return False
    source = str(spec.get("source", "")).strip().lower()
    if source == "fallback_template":
        return True
    if source == "model_generated":
        return False

    intent = self._normalize_adhoc_intent(str(spec.get("intent", "")), default="research")
    slug = self._sanitize_kebab_token(goal, fallback="adhoc-process", max_len=26)
    expected_ids = [
        str(item.get("id", "")).strip()
        for item in self._adhoc_intent_phase_blueprint(intent, slug)
        if isinstance(item, dict) and str(item.get("id", "")).strip()
    ]
    raw_phases = spec.get("phases", [])
    if not isinstance(raw_phases, list) or not raw_phases:
        return False
    observed_ids = [
        str(item.get("id", "")).strip()
        for item in raw_phases
        if isinstance(item, dict) and str(item.get("id", "")).strip()
    ]
    return bool(expected_ids and observed_ids == expected_ids)

def adhoc_synthesis_activity_lines(
    self,
    entry: AdhocProcessCacheEntry,
    *,
    from_cache: bool,
    fresh: bool,
) -> list[str]:
    """Render concise diagnostics for Activity pane visibility."""
    raw_spec = getattr(entry, "spec", None)
    spec = raw_spec if isinstance(raw_spec, dict) else {}
    source = str(spec.get("source", "")).strip() or "unknown"
    intent = self._normalize_adhoc_intent(str(spec.get("intent", "")), default="research")
    phases = spec.get("phases", [])
    phase_count = len(phases) if isinstance(phases, list) else 0
    required = spec.get("required_tools", [])
    recommended = spec.get("recommended_tools", [])
    required_count = len(required) if isinstance(required, list) else 0
    recommended_count = len(recommended) if isinstance(recommended, list) else 0

    lines = [
        (
            "Ad hoc definition summary: "
            f"source={source}, intent={intent}, phases={phase_count}, "
            f"required_tools={required_count}, recommended_tools={recommended_count}."
        ),
        f"Ad hoc cache decision: {'hit' if from_cache else 'miss'} (fresh={fresh}).",
    ]
    synthesis = self._sanitize_synthesis_trace(
        spec.get("_synthesis") if isinstance(spec.get("_synthesis"), dict) else None
    )
    if synthesis:
        initial_ok = bool(synthesis.get("initial_parse_ok"))
        repair_attempted = bool(synthesis.get("repair_attempted"))
        repair_ok = bool(synthesis.get("repair_parse_ok"))
        minimal_attempted = bool(synthesis.get("minimal_retry_attempted"))
        minimal_ok = bool(synthesis.get("minimal_retry_parse_ok"))
        repair_state = "skipped"
        if repair_attempted:
            repair_state = "ok" if repair_ok else "failed"
        minimal_state = "skipped"
        if minimal_attempted:
            minimal_state = "ok" if minimal_ok else "failed"
        response_chars = int(synthesis.get("initial_response_chars") or 0)
        if repair_attempted:
            response_chars = int(synthesis.get("repair_response_chars") or response_chars)
        if minimal_attempted:
            response_chars = int(synthesis.get("minimal_retry_chars") or response_chars)
        lines.append(
            "Ad hoc parse diagnostics: "
            f"initial={'ok' if initial_ok else 'failed'}, "
            f"repair={repair_state}, minimal={minimal_state}, "
            f"response_chars={response_chars}."
        )
        if bool(synthesis.get("empty_response_retry_attempted")):
            retry_chars = int(synthesis.get("empty_response_retry_chars") or 0)
            lines.append(
                f"Ad hoc empty-response retry: attempted, response_chars={retry_chars}.",
            )
        fallback_reason = str(synthesis.get("fallback_reason", "")).strip()
        if fallback_reason:
            lines.append(f"Ad hoc fallback reason: {fallback_reason}.")
        initial_error = str(synthesis.get("initial_error", "")).strip()
        if initial_error:
            lines.append(f"Ad hoc initial error: {initial_error}.")
        repair_error = str(synthesis.get("repair_error", "")).strip()
        if repair_error:
            lines.append(f"Ad hoc repair error: {repair_error}.")
        minimal_error = str(synthesis.get("minimal_retry_error", "")).strip()
        if minimal_error:
            lines.append(f"Ad hoc minimal retry error: {minimal_error}.")
        artifact_dir = str(synthesis.get("artifact_dir", "")).strip()
        if artifact_dir:
            lines.append(f"Ad hoc synthesis artifacts: {artifact_dir}")
        log_path = str(synthesis.get("log_path", "")).strip()
        if log_path:
            lines.append(f"Ad hoc synthesis log: {log_path}")
    return lines

def should_resynthesize_cached_adhoc(self, entry: AdhocProcessCacheEntry) -> bool:
    """Return True when cache should be refreshed from model synthesis."""
    if not self._has_configured_role_model("planner"):
        return False
    spec = entry.spec if isinstance(entry.spec, dict) else {}
    return self._is_template_like_adhoc_spec(spec, goal=entry.goal)

async def synthesize_adhoc_process(self, goal: str, *, key: str) -> AdhocProcessCacheEntry:
    """Generate an ad hoc process definition from a free-form run goal."""
    available_tools = self._available_tool_names()
    fallback_spec = self._fallback_adhoc_spec(
        goal,
        available_tools=available_tools,
        intent="research",
    )
    diagnostics: dict[str, Any] = {
        "cache_key": key,
        "model_name": "",
        "available_tool_count": len(available_tools),
        "goal_chars": len(str(goal or "")),
        "initial_max_tokens": None,
        "repair_max_tokens": None,
        "minimal_max_tokens": None,
        "prompt_chars": 0,
        "initial_response_chars": 0,
        "initial_output_tokens": 0,
        "initial_parse_ok": False,
        "repair_attempted": False,
        "repair_response_chars": 0,
        "repair_output_tokens": 0,
        "repair_parse_ok": False,
        "empty_response_retry_attempted": False,
        "empty_response_retry_chars": 0,
        "minimal_retry_attempted": False,
        "minimal_retry_chars": 0,
        "minimal_output_tokens": 0,
        "minimal_retry_parse_ok": False,
        "parsed_raw_incomplete": False,
        "minimal_retry_kept_prior_raw": False,
        "fallback_reason": "",
        "initial_error": "",
        "repair_error": "",
        "minimal_retry_error": "",
        "initial_preview": "",
        "repair_preview": "",
        "minimal_preview": "",
        "repair_source_chars": 0,
        "repair_source_truncated": False,
        "initial_temperature": None,
        "repair_temperature": None,
        "resolved_source": "fallback_template",
        "resolved_intent": "research",
        "phase_count": len(fallback_spec.get("phases", [])),
        "required_tool_count": len(fallback_spec.get("required_tools", [])),
        "recommended_tool_count": len(fallback_spec.get("recommended_tools", [])),
        "log_path": str(self._adhoc_synthesis_log_path()),
        "artifact_dir": "",
    }
    artifact_dir = self._create_adhoc_synthesis_artifact_dir(key=key, goal=goal)
    if artifact_dir is not None:
        diagnostics["artifact_dir"] = str(artifact_dir)
        self._write_adhoc_synthesis_artifact_text(
            artifact_dir,
            "00-goal.txt",
            str(goal or "").strip(),
        )
        self._write_adhoc_synthesis_artifact_text(
            artifact_dir,
            "00-available-tools.txt",
            "\n".join(available_tools),
        )
    if not self._has_configured_role_model("planner"):
        diagnostics["fallback_reason"] = "model_unavailable"
        self._write_adhoc_synthesis_artifact_text(
            artifact_dir,
            "11-diagnostics.json",
            json.dumps(
                self._sanitize_synthesis_trace(diagnostics),
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
            ),
        )
        self._write_adhoc_synthesis_artifact_text(
            artifact_dir,
            "12-rejection-reason.txt",
            "fallback_reason=model_unavailable\nresolved_source=fallback_template\n",
        )
        fallback_spec["_synthesis"] = self._sanitize_synthesis_trace(diagnostics)
        self._append_adhoc_synthesis_log({
            "event": "adhoc_synthesis",
            "goal": goal,
            **self._sanitize_synthesis_trace(diagnostics),
        })
        logger.warning(
            "Ad hoc synthesis[%s] used fallback: model unavailable",
            key,
        )
        return self._build_adhoc_cache_entry(key=key, goal=goal, spec=fallback_spec)

    tool_list = ", ".join(available_tools) if available_tools else "(none)"
    prompt = (
        "The user wants to do this:\n"
        f"{goal}\n\n"
        "Let's design an abstract Loom process for this request using "
        "docs/creating-packages.md as the contract reference.\n"
        "The process must be reusable, well-structured, and outcome-driven.\n\n"
        "Return ONLY valid JSON with keys:\n"
        "intent, name, description, persona, phase_mode, tool_guidance, required_tools, "
        "recommended_tools, phases.\n"
        "phases must be a list of objects with keys:\n"
        "id, description, depends_on, acceptance_criteria, deliverables.\n\n"
        "Hard requirements:\n"
        "- Determine intent from the user goal and set `intent` to exactly one of: "
        "research, writing, build.\n"
        "- Use lowercase kebab-case for name and phase ids.\n"
        "- phase_mode MUST be one of: strict, guided, suggestive.\n"
        "- Prefer phase_mode=\"guided\" unless strict ordering is explicitly needed.\n"
        "- Use as many phases as needed for the goal (typically 3-20), not a fixed count.\n"
        "- Keep each phase small enough to finish within one subtask wall-clock budget.\n"
        "- Use `depends_on` to model independence so parallelizable phases "
        "can run concurrently.\n"
        "- If intent=build, include an implementation/build phase and a test/verify phase.\n"
        "- If intent=writing, include draft/write and revision/edit phases.\n"
        "- If intent=research, include source/evidence collection and analysis/synthesis "
        "phases.\n"
        "- Every phase must include measurable acceptance_criteria.\n"
        "- Every phase must include concrete deliverable filenames (.md/.csv/etc).\n"
        "- Deliverables should default to root-level filenames (e.g., report.md), "
        "not numbered phase folders, unless the goal explicitly requires "
        "subdirectories.\n"
        "- Do not add phases whose main purpose is creating folder schemas, "
        "workspace schema docs, or numbered directories unless the user "
        "explicitly asks for that structure.\n"
        "- If the goal mentions local files or directories, include a phase that "
        "inspects them.\n"
        "- required_tools must be selected ONLY from available tools.\n"
        "- recommended_tools should list useful missing tools not currently available.\n"
        "- Keep the process reusable and focused on outcomes, not implementation trivia.\n"
        "- Do not wrap JSON in markdown fences.\n\n"
        "Full package authoring reference:\n"
        "<<<BEGIN_CREATING_PACKAGES_MD>>>\n"
        + self._adhoc_package_contract_hint()
        + "\n<<<END_CREATING_PACKAGES_MD>>>\n\n"
        f"Available tools: {tool_list}\n"
    )
    diagnostics["prompt_chars"] = len(prompt)
    self._write_adhoc_synthesis_artifact_text(
        artifact_dir,
        "01-initial-prompt.txt",
        prompt,
    )
    configured_temperature: float | None = None

    expected_json_keys = (
        "intent",
        "name",
        "description",
        "persona",
        "phase_mode",
        "tool_guidance",
        "required_tools",
        "recommended_tools",
        "phases",
    )
    raw: dict[str, Any] | None = None
    raw_text = ""
    try:
        response, model_name, configured_temperature, initial_max_tokens = (
            await self._invoke_helper_role_completion(
                role="planner",
                tier=2,
                prompt=prompt,
                max_tokens=None,
            )
        )
        diagnostics["model_name"] = model_name
        diagnostics["initial_temperature"] = configured_temperature
        diagnostics["repair_temperature"] = configured_temperature
        diagnostics["initial_max_tokens"] = initial_max_tokens
        raw_text = str(getattr(response, "text", "") or "")
        diagnostics["initial_response_chars"] = len(raw_text)
        diagnostics["initial_preview"] = self._synthesis_preview(raw_text)
        self._write_adhoc_synthesis_artifact_text(
            artifact_dir,
            "02-initial-response.txt",
            raw_text,
        )
        raw = self._extract_json_payload(
            raw_text,
            expected_keys=expected_json_keys,
        )
        diagnostics["initial_parse_ok"] = isinstance(raw, dict)
    except Exception as e:
        diagnostics["initial_error"] = str(e)
        self._write_adhoc_synthesis_artifact_text(
            artifact_dir,
            "02-initial-error.txt",
            str(e),
        )
        logger.warning("Ad hoc process synthesis failed: %s", e)

    if raw is None and not raw_text.strip() and not str(
        diagnostics.get("initial_error", ""),
    ).strip():
        diagnostics["repair_attempted"] = True
        diagnostics["empty_response_retry_attempted"] = True
        retry_prompt = (
            "Your previous response was empty.\n"
            "Return a non-empty strict JSON object only.\n"
            "Required keys: intent, name, description, persona, phase_mode, "
            "tool_guidance, required_tools, recommended_tools, phases.\n"
            "Each phase object keys: id, description, depends_on, "
            "acceptance_criteria, deliverables.\n\n"
            "Deliverables should default to root-level filenames (e.g., "
            "report.md) unless the user explicitly requires subdirectories.\n"
            "Do not include folder-scaffolding-only phases unless explicitly requested.\n\n"
            "Use this same task request:\n"
            f"{goal}\n\n"
            "Do not include markdown fences."
        )
        self._write_adhoc_synthesis_artifact_text(
            artifact_dir,
            "03-empty-retry-prompt.txt",
            retry_prompt,
        )
        try:
            retry_response, retry_model_name, retry_temperature, retry_max_tokens = (
                await self._invoke_helper_role_completion(
                    role="planner",
                    tier=2,
                    prompt=retry_prompt,
                    max_tokens=None,
                    temperature=configured_temperature,
                )
            )
            if not diagnostics["model_name"]:
                diagnostics["model_name"] = retry_model_name
            if configured_temperature is None:
                configured_temperature = retry_temperature
                diagnostics["repair_temperature"] = retry_temperature
            diagnostics["repair_max_tokens"] = retry_max_tokens
            retry_text = str(getattr(retry_response, "text", "") or "")
            diagnostics["empty_response_retry_chars"] = len(retry_text)
            diagnostics["repair_response_chars"] = len(retry_text)
            diagnostics["repair_preview"] = self._synthesis_preview(retry_text)
            self._write_adhoc_synthesis_artifact_text(
                artifact_dir,
                "04-empty-retry-response.txt",
                retry_text,
            )
            if retry_text.strip():
                raw_text = retry_text
                raw = self._extract_json_payload(
                    retry_text,
                    expected_keys=expected_json_keys,
                )
                diagnostics["repair_parse_ok"] = isinstance(raw, dict)
        except Exception as e:
            diagnostics["repair_error"] = str(e)
            self._write_adhoc_synthesis_artifact_text(
                artifact_dir,
                "04-empty-retry-error.txt",
                str(e),
            )
            logger.warning("Ad hoc process empty-response retry failed: %s", e)

    if raw is None and raw_text.strip():
        diagnostics["repair_attempted"] = True
        repair_source_cap = int(
            getattr(
                getattr(self._config, "limits", None),
                "adhoc_repair_source_max_chars",
                0,
            )
            or 0,
        )
        repair_source_text = raw_text
        if repair_source_cap > 0 and len(repair_source_text) > repair_source_cap:
            repair_source_text = repair_source_text[:repair_source_cap]
        diagnostics["repair_source_chars"] = len(repair_source_text)
        diagnostics["repair_source_truncated"] = len(repair_source_text) < len(raw_text)
        repair_prompt = (
            "You will receive model output that should describe a Loom process.\n"
            "Convert it into STRICT JSON only.\n"
            "Return exactly one JSON object with keys:\n"
            "intent, name, description, persona, phase_mode, tool_guidance, "
            "required_tools, recommended_tools, phases.\n"
            "Each phase object must contain: id, description, depends_on, "
            "acceptance_criteria, deliverables.\n"
            "Prefer root-level deliverable filenames unless the user explicitly "
            "requires subdirectories.\n"
            "Do not include folder-scaffolding-only phases unless explicitly requested.\n"
            "Do not include markdown fences.\n\n"
            "SOURCE OUTPUT:\n"
            "<<<BEGIN_SOURCE>>>\n"
            f"{repair_source_text}\n"
            "<<<END_SOURCE>>>"
        )
        self._write_adhoc_synthesis_artifact_text(
            artifact_dir,
            "05-repair-prompt.txt",
            repair_prompt,
        )
        try:
            repaired, repaired_model_name, repaired_temperature, repair_max_tokens = (
                await self._invoke_helper_role_completion(
                    role="planner",
                    tier=2,
                    prompt=repair_prompt,
                    max_tokens=None,
                    temperature=configured_temperature,
                )
            )
            if not diagnostics["model_name"]:
                diagnostics["model_name"] = repaired_model_name
            if configured_temperature is None:
                configured_temperature = repaired_temperature
                diagnostics["repair_temperature"] = repaired_temperature
            diagnostics["repair_max_tokens"] = repair_max_tokens
            repaired_text = str(getattr(repaired, "text", "") or "")
            diagnostics["repair_response_chars"] = len(repaired_text)
            diagnostics["repair_preview"] = self._synthesis_preview(repaired_text)
            self._write_adhoc_synthesis_artifact_text(
                artifact_dir,
                "06-repair-response.txt",
                repaired_text,
            )
            raw = self._extract_json_payload(
                repaired_text,
                expected_keys=expected_json_keys,
            )
            diagnostics["repair_parse_ok"] = isinstance(raw, dict)
        except Exception as e:
            diagnostics["repair_error"] = str(e)
            self._write_adhoc_synthesis_artifact_text(
                artifact_dir,
                "06-repair-error.txt",
                str(e),
            )
            logger.warning("Ad hoc process JSON repair failed: %s", e)

    parsed_raw = raw if isinstance(raw, dict) else None
    needs_minimal_retry = (
        raw is None or self._raw_adhoc_spec_needs_minimal_retry(parsed_raw)
    )
    diagnostics["parsed_raw_incomplete"] = bool(
        isinstance(parsed_raw, dict)
        and self._raw_adhoc_spec_needs_minimal_retry(parsed_raw),
    )
    if needs_minimal_retry:
        diagnostics["minimal_retry_attempted"] = True
        minimal_prompt = (
            "Return exactly one STRICT JSON object and nothing else.\n"
            "Required top-level keys:\n"
            "intent, name, description, persona, phase_mode, tool_guidance, "
            "required_tools, recommended_tools, phases.\n"
            "Constraints:\n"
            "- intent: research | writing | build\n"
            "- phase_mode: strict | guided | suggestive (prefer guided)\n"
            "- phases length: 3-20\n"
            "- each phase object keys: id, description, depends_on, "
            "acceptance_criteria, deliverables\n"
            "- required_tools must be from available tools only\n"
            "- description, persona, tool_guidance must be concise (<= 180 chars each)\n"
            "- phase descriptions must be concise (<= 120 chars)\n"
            "- acceptance_criteria must be a short STRING (not array)\n"
            "- deliverables must be filename strings like report.md\n"
            "- prefer root-level deliverables; avoid numbered phase folders unless "
            "explicitly required by the goal\n"
            "- do not include folder-scaffolding-only phases unless explicitly requested\n"
            "- do not include markdown/code fences/prose\n\n"
            f"Goal:\n{goal}\n\n"
            f"Available tools: {tool_list}\n"
        )
        self._write_adhoc_synthesis_artifact_text(
            artifact_dir,
            "07-minimal-retry-prompt.txt",
            minimal_prompt,
        )
        try:
            minimal, minimal_model_name, minimal_temperature, minimal_max_tokens = (
                await self._invoke_helper_role_completion(
                    role="planner",
                    tier=2,
                    prompt=minimal_prompt,
                    max_tokens=None,
                    temperature=configured_temperature,
                )
            )
            if not diagnostics["model_name"]:
                diagnostics["model_name"] = minimal_model_name
            if configured_temperature is None:
                configured_temperature = minimal_temperature
                diagnostics["repair_temperature"] = minimal_temperature
            diagnostics["minimal_max_tokens"] = minimal_max_tokens
            minimal_text = str(getattr(minimal, "text", "") or "")
            diagnostics["minimal_retry_chars"] = len(minimal_text)
            diagnostics["minimal_preview"] = self._synthesis_preview(minimal_text)
            self._write_adhoc_synthesis_artifact_text(
                artifact_dir,
                "08-minimal-retry-response.txt",
                minimal_text,
            )
            minimal_parsed = self._extract_json_payload(
                minimal_text,
                expected_keys=expected_json_keys,
            )
            diagnostics["minimal_retry_parse_ok"] = isinstance(minimal_parsed, dict)
            if isinstance(minimal_parsed, dict):
                raw = minimal_parsed
            elif isinstance(parsed_raw, dict):
                # Preserve previously parsed content when minimal retry fails.
                diagnostics["minimal_retry_kept_prior_raw"] = True
                raw = parsed_raw
        except Exception as e:
            diagnostics["minimal_retry_error"] = str(e)
            self._write_adhoc_synthesis_artifact_text(
                artifact_dir,
                "08-minimal-retry-error.txt",
                str(e),
            )
            if isinstance(parsed_raw, dict):
                diagnostics["minimal_retry_kept_prior_raw"] = True
                raw = parsed_raw
            logger.warning("Ad hoc process minimal retry failed: %s", e)

    if raw is None:
        if self._is_temperature_one_only_error(diagnostics.get("initial_error", "")):
            diagnostics["fallback_reason"] = "temperature_config_mismatch"
        elif self._is_temperature_one_only_error(diagnostics.get("repair_error", "")):
            diagnostics["fallback_reason"] = "temperature_config_mismatch"
        elif not raw_text.strip():
            diagnostics["fallback_reason"] = "empty_model_response"
        elif str(diagnostics.get("initial_error", "")).strip():
            diagnostics["fallback_reason"] = "model_completion_error"
        elif (
            diagnostics.get("repair_attempted")
            and not diagnostics.get("repair_parse_ok")
            and diagnostics.get("minimal_retry_attempted")
            and not diagnostics.get("minimal_retry_parse_ok")
        ):
            diagnostics["fallback_reason"] = "schema_parse_failed"
        else:
            diagnostics["fallback_reason"] = "non_parseable_response"
        preview = re.sub(r"\s+", " ", raw_text).strip()
        if len(preview) > 280:
            preview = preview[:280].rstrip() + "..."
        logger.warning(
            "Ad hoc process synthesis returned non-parseable payload; using fallback."
            " preview=%r",
            preview,
        )

    normalized = self._normalize_adhoc_spec(
        raw,
        goal=goal,
        key=key,
        available_tools=available_tools,
    )
    source = str(normalized.get("source", "")).strip() or "unknown"
    diagnostics["resolved_source"] = source
    diagnostics["resolved_intent"] = str(normalized.get("intent", "")).strip() or "research"
    diagnostics["phase_count"] = len(normalized.get("phases", []) or [])
    diagnostics["required_tool_count"] = len(normalized.get("required_tools", []) or [])
    diagnostics["recommended_tool_count"] = len(normalized.get("recommended_tools", []) or [])
    if (
        source == "fallback_template"
        and not str(diagnostics.get("fallback_reason", "")).strip()
    ):
        diagnostics["fallback_reason"] = "normalization_template_substitution"

    self._write_adhoc_synthesis_artifact_yaml(
        artifact_dir,
        "09-parsed-raw.yaml",
        raw if isinstance(raw, dict) else None,
    )
    self._write_adhoc_synthesis_artifact_yaml(
        artifact_dir,
        "10-normalized-spec.yaml",
        normalized,
    )
    self._write_adhoc_synthesis_artifact_text(
        artifact_dir,
        "11-diagnostics.json",
        json.dumps(
            self._sanitize_synthesis_trace(diagnostics),
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        ),
    )
    if str(diagnostics.get("fallback_reason", "")).strip():
        self._write_adhoc_synthesis_artifact_text(
            artifact_dir,
            "12-rejection-reason.txt",
            (
                "fallback_reason="
                f"{str(diagnostics.get('fallback_reason', '')).strip()}\n"
                "resolved_source="
                f"{str(diagnostics.get('resolved_source', '')).strip()}\n"
                "parsed_raw_incomplete="
                f"{bool(diagnostics.get('parsed_raw_incomplete'))}\n"
            ),
        )

    normalized["_synthesis"] = self._sanitize_synthesis_trace(diagnostics)
    self._append_adhoc_synthesis_log({
        "event": "adhoc_synthesis",
        "goal": goal,
        **self._sanitize_synthesis_trace(diagnostics),
    })
    logger.warning(
        "Ad hoc synthesis[%s]: source=%s intent=%s phases=%s parse(initial=%s repair=%s)",
        key,
        diagnostics["resolved_source"],
        diagnostics["resolved_intent"],
        diagnostics["phase_count"],
        diagnostics["initial_parse_ok"],
        diagnostics["repair_parse_ok"] if diagnostics["repair_attempted"] else "skipped",
    )
    return self._build_adhoc_cache_entry(key=key, goal=goal, spec=normalized)

async def get_or_create_adhoc_process(
    self,
    goal: str,
    *,
    fresh: bool = False,
) -> tuple[AdhocProcessCacheEntry, bool]:
    """Fetch cached ad hoc process for goal, or synthesize and cache one."""
    key = self._adhoc_cache_key(goal)
    if fresh:
        # Explicit --fresh requests should always bypass memory + disk cache.
        self._adhoc_process_cache.pop(key, None)
        generated = await self._synthesize_adhoc_process(goal, key=key)
        self._adhoc_process_cache[key] = generated
        try:
            self._persist_adhoc_cache_entry(generated)
        except Exception as e:
            logger.warning("Failed to persist ad hoc process cache for %s: %s", key, e)
        self._append_adhoc_synthesis_log({
            "event": "adhoc_cache_decision",
            "cache_key": key,
            "goal": goal,
            "decision": "fresh_synthesis",
        })
        return generated, False

    cached = self._adhoc_process_cache.get(key)
    if cached is not None and not self._should_resynthesize_cached_adhoc(cached):
        self._append_adhoc_synthesis_log({
            "event": "adhoc_cache_decision",
            "cache_key": key,
            "goal": goal,
            "decision": "memory_hit",
        })
        return cached, True
    if cached is not None:
        self._adhoc_process_cache.pop(key, None)
    disk_cached = self._load_adhoc_cache_entry_from_disk(key)
    if disk_cached is not None and not self._should_resynthesize_cached_adhoc(disk_cached):
        self._adhoc_process_cache[key] = disk_cached
        self._append_adhoc_synthesis_log({
            "event": "adhoc_cache_decision",
            "cache_key": key,
            "goal": goal,
            "decision": "disk_hit",
        })
        return disk_cached, True
    generated = await self._synthesize_adhoc_process(goal, key=key)
    self._adhoc_process_cache[key] = generated
    try:
        self._persist_adhoc_cache_entry(generated)
    except Exception as e:
        logger.warning("Failed to persist ad hoc process cache for %s: %s", key, e)
    self._append_adhoc_synthesis_log({
        "event": "adhoc_cache_decision",
        "cache_key": key,
        "goal": goal,
        "decision": "synthesized",
    })
    return generated, False


def serialize_process_for_package(process_defn: ProcessDefinition) -> dict[str, Any]:
    """Convert a process definition into a process.yaml payload."""
    payload: dict[str, Any] = {
        "name": process_defn.name,
        "schema_version": int(process_defn.schema_version or 1),
        "version": __version__,
        "description": process_defn.description,
        "persona": process_defn.persona,
        "tool_guidance": process_defn.tool_guidance,
        "phase_mode": process_defn.phase_mode or "guided",
        "tools": {
            "guidance": process_defn.tools.guidance,
            "required": list(process_defn.tools.required),
            "excluded": list(process_defn.tools.excluded),
        },
        "phases": [
            {
                "id": phase.id,
                "description": phase.description,
                "depends_on": list(phase.depends_on),
                "acceptance_criteria": phase.acceptance_criteria,
                "deliverables": list(phase.deliverables),
            }
            for phase in process_defn.phases
        ],
        "tags": list(dict.fromkeys([*process_defn.tags, "adhoc", "generated"])),
        "author": process_defn.author or "loom-adhoc",
    }
    auth_required = process_defn.auth_required_resources()
    if auth_required:
        payload["auth"] = {"required": auth_required}
    return payload

def save_adhoc_process_package(
    self,
    *,
    process_defn: ProcessDefinition,
    package_name: str,
    recommended_tools: list[str],
) -> Path:
    """Persist an ad hoc process as a workspace-local package."""
    import yaml

    safe_name = self._sanitize_kebab_token(
        package_name,
        fallback="adhoc-process",
        max_len=40,
    )
    package_dir = self._workspace / ".loom" / "processes" / safe_name
    if package_dir.exists():
        raise ValueError(f"Package already exists: {package_dir}")
    package_dir.mkdir(parents=True, exist_ok=False)

    spec = self._serialize_process_for_package(process_defn)
    spec["name"] = safe_name
    process_yaml = package_dir / "process.yaml"
    process_yaml.write_text(
        yaml.safe_dump(spec, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    notes = [
        f"# {safe_name}",
        "",
        "Generated from an ad hoc `/run` synthesis in Loom TUI.",
    ]
    if recommended_tools:
        notes.append("")
        notes.append("## Recommended Additional Tools")
        for tool_name in recommended_tools:
            notes.append(f"- {tool_name}")
    (package_dir / "README.md").write_text(
        "\n".join(notes).rstrip() + "\n",
        encoding="utf-8",
    )
    return package_dir
