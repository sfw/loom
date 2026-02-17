"""Engine lifecycle: wires up all Loom components for the API server."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from loom.config import Config
from loom.engine.orchestrator import Orchestrator
from loom.events.bus import EventBus, EventPersister
from loom.events.webhook import WebhookDelivery
from loom.learning.manager import LearningManager
from loom.models.router import ModelRouter
from loom.prompts.assembler import PromptAssembler
from loom.recovery.approval import ApprovalManager
from loom.state.memory import Database, MemoryManager
from loom.state.task_state import TaskStateManager
from loom.tools import create_default_registry
from loom.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from loom.processes.schema import ProcessDefinition


class Engine:
    """Holds all Loom components. Created during server lifespan."""

    def __init__(
        self,
        config: Config,
        orchestrator: Orchestrator,
        event_bus: EventBus,
        model_router: ModelRouter,
        tool_registry: ToolRegistry,
        memory_manager: MemoryManager,
        state_manager: TaskStateManager,
        prompt_assembler: PromptAssembler,
        database: Database,
        approval_manager: ApprovalManager,
        webhook_delivery: WebhookDelivery,
        learning_manager: LearningManager,
    ):
        self.config = config
        self.orchestrator = orchestrator
        self.event_bus = event_bus
        self.model_router = model_router
        self.tool_registry = tool_registry
        self.memory_manager = memory_manager
        self.state_manager = state_manager
        self.prompt_assembler = prompt_assembler
        self.database = database
        self.approval_manager = approval_manager
        self.webhook_delivery = webhook_delivery
        self.learning_manager = learning_manager

    async def shutdown(self) -> None:
        """Graceful cleanup."""
        await self.model_router.close()
        await self.database.close()

    def create_task_orchestrator(
        self, process: ProcessDefinition | None = None,
    ) -> Orchestrator:
        """Create an isolated orchestrator for a single task execution.

        A fresh prompt assembler and tool registry avoid cross-task mutation
        when a process definition applies tool exclusions or prompt overrides.
        """
        return Orchestrator(
            model_router=self.model_router,
            tool_registry=create_default_registry(self.config),
            memory_manager=self.memory_manager,
            prompt_assembler=PromptAssembler(),
            state_manager=self.state_manager,
            event_bus=self.event_bus,
            config=self.config,
            approval_manager=self.approval_manager,
            learning_manager=self.learning_manager,
            process=process,
        )


async def create_engine(config: Config) -> Engine:
    """Create and initialize all Loom components."""
    import logging

    logger = logging.getLogger("loom.engine")

    # Database
    db_path = Path(config.memory.database_path).expanduser()
    try:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        database = Database(str(db_path))
        await database.initialize()
    except Exception as e:
        logger.error("Database initialization failed: %s", e)
        raise RuntimeError(
            f"Cannot initialize database at {db_path}: {e}"
        ) from e

    # State manager
    data_dir = Path(config.workspace.scratch_dir).expanduser()
    try:
        data_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error("Cannot create data directory: %s", e)
        raise RuntimeError(
            f"Cannot create data directory {data_dir}: {e}"
        ) from e
    state_manager = TaskStateManager(data_dir=data_dir)

    # Memory
    memory_manager = MemoryManager(database)

    # Models
    model_router = ModelRouter.from_config(config)

    # Tools
    tool_registry = create_default_registry(config)

    # Prompts
    prompt_assembler = PromptAssembler()

    # Events
    event_bus = EventBus()

    # Event persistence — subscribe to all events and write to SQLite
    event_persister = EventPersister(database)
    event_persister.attach(event_bus)

    # Webhook delivery — subscribe to terminal events
    webhook_delivery = WebhookDelivery()
    webhook_delivery.attach(event_bus)

    # Approval manager
    approval_manager = ApprovalManager(event_bus)

    # Learning manager
    learning_manager = LearningManager(database)

    # Orchestrator
    orchestrator = Orchestrator(
        model_router=model_router,
        tool_registry=tool_registry,
        memory_manager=memory_manager,
        prompt_assembler=prompt_assembler,
        state_manager=state_manager,
        event_bus=event_bus,
        config=config,
        approval_manager=approval_manager,
        learning_manager=learning_manager,
    )

    return Engine(
        config=config,
        orchestrator=orchestrator,
        event_bus=event_bus,
        model_router=model_router,
        tool_registry=tool_registry,
        memory_manager=memory_manager,
        state_manager=state_manager,
        prompt_assembler=prompt_assembler,
        database=database,
        approval_manager=approval_manager,
        webhook_delivery=webhook_delivery,
        learning_manager=learning_manager,
    )
