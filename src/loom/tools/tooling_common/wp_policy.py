"""WordPress destructive-operation risk detection and policy helpers."""

from __future__ import annotations

import shlex
from dataclasses import dataclass


@dataclass(frozen=True)
class WPRiskAssessment:
    """High-level risk classification for WP operations."""

    high_risk: bool
    risk_level: str
    action_class: str
    impact_preview: str
    consequences: str


HIGH_RISK_ACTIONS = frozenset({
    ("db", "reset"),
    ("db", "drop"),
    ("plugin", "delete"),
    ("theme", "delete"),
})


def assess_wp_cli_risk(
    group: str,
    action: str,
    args: dict | None = None,
) -> WPRiskAssessment | None:
    """Return a high-risk assessment for wp_cli call when applicable."""
    clean_group = str(group or "").strip().lower()
    clean_action = str(action or "").strip().lower()
    payload = args if isinstance(args, dict) else {}

    if (clean_group, clean_action) in HIGH_RISK_ACTIONS:
        impact = f"wp {clean_group} {clean_action}"
        return WPRiskAssessment(
            high_risk=True,
            risk_level="high",
            action_class="destructive-wordpress-operation",
            impact_preview=impact,
            consequences=(
                "This operation can permanently delete WordPress database data "
                "or assets. "
                "Ensure backups/rollback strategy exist before approval."
            ),
        )

    if clean_group == "search_replace":
        dry_run = bool(payload.get("dry_run", True))
        all_tables = bool(payload.get("all_tables", False))
        if (not dry_run) and all_tables:
            return WPRiskAssessment(
                high_risk=True,
                risk_level="high",
                action_class="destructive-database-rewrite",
                impact_preview=(
                    "site-wide search-replace on all database tables "
                    "(non-dry-run)"
                ),
                consequences=(
                    "This can irreversibly rewrite large portions of site content/config. "
                    "Verify match scope and backups before approval."
                ),
            )

    return None


def assess_wp_shell_command_risk(command: str) -> WPRiskAssessment | None:
    """Detect high-risk WordPress CLI usage inside shell_execute commands."""
    text = str(command or "").strip()
    if not text:
        return None

    try:
        tokens = shlex.split(text)
    except ValueError:
        return None
    if not tokens:
        return None

    # Account for wrappers like: php wp-cli.phar ...
    if tokens[0] == "php" and len(tokens) >= 2 and "wp" in tokens[1]:
        tokens = ["wp", *tokens[2:]]

    # Simple guard: only inspect explicit wp invocations.
    if tokens[0] != "wp":
        return None

    # Strip option flags to find command nouns.
    words = [tok for tok in tokens[1:] if tok and not tok.startswith("-")]
    if not words:
        return None

    group = words[0].lower()
    action = words[1].lower() if len(words) > 1 else ""

    shell_args = {
        "dry_run": "--dry-run" in tokens,
        "all_tables": "--all-tables" in tokens or "--all-tables-with-prefix" in tokens,
    }
    return assess_wp_cli_risk(group, action, shell_args)


def format_wp_risk_info(assessment: WPRiskAssessment) -> dict:
    """Serialize risk assessment for approval/modal display payloads."""
    return {
        "risk_level": assessment.risk_level,
        "action_class": assessment.action_class,
        "impact_preview": assessment.impact_preview,
        "consequences": assessment.consequences,
    }
