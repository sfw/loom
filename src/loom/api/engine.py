"""Engine lifecycle: wires up all Loom components for the API server."""

from __future__ import annotations

from pathlib import Path

from loom.config import Config
from loom.engine.orchestrator import Orchestrator
from loom.events.bus import EventBus
from loom.models.router import ModelRouter
from loom.prompts.assembler import PromptAssembler
from loom.state.memory import Database, MemoryManager
from loom.state.task_state import TaskStateManager
from loom.tools import create_default_registry
from loom.tools.registry import ToolRegistry


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

    async def shutdown(self) -> None:
        """Graceful cleanup."""
        await self.model_router.close()
        await self.database.close()


async def create_engine(config: Config) -> Engine:
    """Create and initialize all Loom components."""
    # Database
    db_path = Path(config.memory.database_path).expanduser()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    database = Database(str(db_path))
    await database.initialize()

    # State manager
    data_dir = Path(config.workspace.scratch_dir).expanduser()
    data_dir.mkdir(parents=True, exist_ok=True)
    state_manager = TaskStateManager(data_dir=data_dir)

    # Memory
    memory_manager = MemoryManager(database)

    # Models
    model_router = ModelRouter.from_config(config)

    # Tools
    tool_registry = create_default_registry()

    # Prompts
    prompt_assembler = PromptAssembler()

    # Events
    event_bus = EventBus()

    # Orchestrator
    orchestrator = Orchestrator(
        model_router=model_router,
        tool_registry=tool_registry,
        memory_manager=memory_manager,
        prompt_assembler=prompt_assembler,
        state_manager=state_manager,
        event_bus=event_bus,
        config=config,
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
    )
