"""Persistence/delegate tool binding helpers."""

from __future__ import annotations

import logging
from pathlib import Path

from loom.processes.schema import ProcessDefinition

logger = logging.getLogger("loom.tui.app.core")


def ensure_persistence_tools(self) -> None:
    """Ensure recall/delegate tools are registered and tracked."""
    self._recall_tool = None
    self._delegate_tool = None

    from loom.tools.conversation_recall import ConversationRecallTool
    from loom.tools.delegate_task import DelegateTaskTool

    if self._store is not None:
        recall = self._tools.get("conversation_recall")
        if recall is not None and not isinstance(recall, ConversationRecallTool):
            logger.warning(
                "Replacing unexpected conversation_recall tool type: %s",
                type(recall).__name__,
            )
            self._tools.exclude("conversation_recall")
            recall = None
        if recall is None:
            recall = ConversationRecallTool()
            self._tools.register(recall)
        self._recall_tool = recall

    delegate = self._tools.get("delegate_task")
    if delegate is not None and not isinstance(delegate, DelegateTaskTool):
        logger.warning(
            "Replacing unexpected delegate_task tool type: %s",
            type(delegate).__name__,
        )
        self._tools.exclude("delegate_task")
        delegate = None
    if delegate is None:
        delegate = DelegateTaskTool()
        self._tools.register(delegate)
    self._delegate_tool = delegate


def ensure_delegate_task_ready_for_run(self) -> tuple[bool, str]:
    """Best-effort rebind of delegate_task before `/run` launch."""
    if self._session is not None and self._config is not None and self._db is not None:
        self._ensure_persistence_tools()
        self._bind_session_tools()
    delegate = self._tools.get("delegate_task")
    if delegate is None:
        return False, "delegate_task tool is missing."
    factory = getattr(delegate, "_factory", None)
    if (
        self._session is not None
        and self._config is not None
        and self._db is not None
        and not callable(factory)
    ):
        return False, "delegate_task is unbound (no orchestrator configured)."
    return True, ""


def bind_session_tools(self) -> None:
    """Bind tools that hold a reference to the active session."""
    if self._session is None:
        return
    if self._recall_tool and self._store:
        self._recall_tool.bind(
            store=self._store,
            session_id=self._session.session_id,
            session_state=self._session.session_state,
            compactor=getattr(self._session, "compactor", None),
            v2_actions_enabled=self._cowork_memory_index_v2_actions_enabled(),
            force_fts=self._cowork_memory_index_force_fts(),
        )
    if self._delegate_tool and self._config and self._db:
        try:
            from loom.engine.orchestrator import Orchestrator
            from loom.events.bus import EventBus
            from loom.models.router import ModelRouter
            from loom.prompts.assembler import PromptAssembler
            from loom.state.memory import MemoryManager
            from loom.state.task_state import TaskStateManager
            from loom.tools import create_default_registry as _create_tools

            config = self._config
            db = self._db

            if hasattr(config, "workspace"):
                data_dir = Path(
                    config.workspace.scratch_dir,
                ).expanduser()
            else:
                data_dir = Path.home() / ".loom"

            router = ModelRouter.from_config(config)

            async def _orchestrator_factory(
                process_override: ProcessDefinition | None = None,
            ):
                telemetry_cfg = getattr(config, "telemetry", None)
                event_bus_kwargs = {
                    "debug_diagnostics_rate_per_minute": int(
                        getattr(
                            telemetry_cfg,
                            "debug_diagnostics_rate_per_minute",
                            120,
                        ),
                    ),
                    "debug_diagnostics_burst": int(
                        getattr(telemetry_cfg, "debug_diagnostics_burst", 30),
                    ),
                }
                try:
                    event_bus = EventBus(**event_bus_kwargs)
                except TypeError:
                    event_bus = EventBus()
                event_bus.set_operator_mode_provider(self._effective_telemetry_mode)
                return Orchestrator(
                    model_router=router,
                    tool_registry=_create_tools(config),
                    memory_manager=MemoryManager(db),
                    prompt_assembler=PromptAssembler(),
                    state_manager=TaskStateManager(data_dir),
                    event_bus=event_bus,
                    config=config,
                    process=process_override or self._process_defn,
                )

            self._delegate_tool.bind(_orchestrator_factory)
        except Exception as e:
            logger.warning("Failed to bind delegate_task tool: %s", e)
