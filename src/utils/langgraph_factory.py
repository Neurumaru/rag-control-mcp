"""LangGraph utilities and factory functions for MCP-RAG-Control system."""

import asyncio
from contextlib import asynccontextmanager, contextmanager
from typing import Any, Dict, Optional, Union

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.redis import RedisSaver
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from .config import LangGraphConfig, get_config
from .logger import get_logger, log_langgraph_checkpoint_event

logger = get_logger(__name__)


class LangGraphCheckpointerFactory:
    """Factory for creating LangGraph checkpointers based on configuration."""

    @staticmethod
    def create_checkpointer(
        config: Optional[LangGraphConfig] = None, async_mode: bool = False
    ) -> BaseCheckpointSaver:
        """Create a checkpointer based on configuration.

        Args:
            config: LangGraph configuration. If None, uses global config.
            async_mode: Whether to create async or sync checkpointer

        Returns:
            Configured checkpointer instance

        Raises:
            ValueError: If configuration is invalid
            RuntimeError: If checkpointer creation fails
        """
        if config is None:
            app_config = get_config()
            config = app_config.langgraph

        checkpointer_type = config.checkpointer_type.lower()

        try:
            if checkpointer_type == "memory":
                return MemorySaver()

            elif checkpointer_type == "postgres":
                if not config.checkpoint_db_uri:
                    raise ValueError("PostgreSQL checkpointer requires checkpoint_db_uri")

                if async_mode:
                    return AsyncPostgresSaver.from_conn_string(config.checkpoint_db_uri)
                else:
                    return PostgresSaver.from_conn_string(config.checkpoint_db_uri)

            elif checkpointer_type == "redis":
                if not config.checkpoint_db_uri:
                    raise ValueError("Redis checkpointer requires checkpoint_db_uri")

                if async_mode:
                    return AsyncRedisSaver.from_conn_string(config.checkpoint_db_uri)
                else:
                    return RedisSaver.from_conn_string(config.checkpoint_db_uri)

            elif checkpointer_type == "sqlite":
                # Use in-memory SQLite if no URI provided
                db_uri = config.checkpoint_db_uri or ":memory:"

                if async_mode:
                    return AsyncSqliteSaver.from_conn_string(db_uri)
                else:
                    return SqliteSaver.from_conn_string(db_uri)

            else:
                raise ValueError(f"Unsupported checkpointer type: {checkpointer_type}")

        except Exception as e:
            logger.error(f"Failed to create {checkpointer_type} checkpointer: {e}")
            raise RuntimeError(f"Checkpointer creation failed: {e}") from e


class LangGraphConfigManager:
    """Manager for LangGraph runtime configurations."""

    @staticmethod
    def create_thread_config(
        thread_id: str,
        user_id: Optional[str] = None,
        checkpoint_id: Optional[str] = None,
        **extra_config,
    ) -> Dict[str, Any]:
        """Create LangGraph thread configuration.

        Args:
            thread_id: Unique thread identifier
            user_id: Optional user identifier
            checkpoint_id: Optional specific checkpoint to resume from
            **extra_config: Additional configuration parameters

        Returns:
            Configuration dictionary for LangGraph
        """
        config = {"configurable": {"thread_id": thread_id, **extra_config}}

        if user_id:
            config["configurable"]["user_id"] = user_id

        if checkpoint_id:
            config["configurable"]["checkpoint_id"] = checkpoint_id

        logger.debug(f"Created thread config for thread_id: {thread_id}")
        return config

    @staticmethod
    def create_execution_config(
        stream_mode: str = "values",
        recursion_limit: Optional[int] = None,
        debug: bool = False,
    ) -> Dict[str, Any]:
        """Create LangGraph execution configuration.

        Args:
            stream_mode: Stream mode for graph execution
            recursion_limit: Maximum recursion depth
            debug: Enable debug mode

        Returns:
            Execution configuration dictionary
        """
        app_config = get_config()
        lg_config = app_config.langgraph

        config = {
            "stream_mode": stream_mode or lg_config.stream_mode,
            "debug": debug or lg_config.enable_debug_mode,
        }

        if recursion_limit is not None:
            config["recursion_limit"] = recursion_limit
        elif lg_config.max_recursion_limit:
            config["recursion_limit"] = lg_config.max_recursion_limit

        return config


class LangGraphMonitor:
    """Monitor for LangGraph execution metrics and events."""

    def __init__(self, thread_id: str):
        """Initialize monitor for a specific thread.

        Args:
            thread_id: Thread ID to monitor
        """
        self.thread_id = thread_id
        self.execution_metrics = {}
        self.checkpoints_created = 0
        self.checkpoints_restored = 0

    def record_node_execution(
        self,
        node_name: str,
        duration_ms: float,
        success: bool = True,
        error: Optional[Exception] = None,
    ) -> None:
        """Record node execution metrics.

        Args:
            node_name: Name of the executed node
            duration_ms: Execution duration in milliseconds
            success: Whether execution succeeded
            error: Exception if execution failed
        """
        if node_name not in self.execution_metrics:
            self.execution_metrics[node_name] = {
                "total_executions": 0,
                "total_duration_ms": 0,
                "successful_executions": 0,
                "failed_executions": 0,
                "avg_duration_ms": 0,
            }

        metrics = self.execution_metrics[node_name]
        metrics["total_executions"] += 1
        metrics["total_duration_ms"] += duration_ms

        if success:
            metrics["successful_executions"] += 1
        else:
            metrics["failed_executions"] += 1

        metrics["avg_duration_ms"] = metrics["total_duration_ms"] / metrics["total_executions"]

        logger.debug(
            f"Node execution recorded: {node_name} "
            f"(success: {success}, duration: {duration_ms}ms)"
        )

    def record_checkpoint_event(self, event_type: str, checkpoint_id: str) -> None:
        """Record checkpoint events.

        Args:
            event_type: Type of checkpoint event
            checkpoint_id: Checkpoint ID
        """
        if event_type == "created":
            self.checkpoints_created += 1
        elif event_type == "restored":
            self.checkpoints_restored += 1

        log_langgraph_checkpoint_event(event_type, self.thread_id, checkpoint_id)

    def get_execution_summary(self) -> Dict[str, Any]:
        """Get execution summary for the monitored thread.

        Returns:
            Dictionary containing execution metrics summary
        """
        total_executions = sum(
            metrics["total_executions"] for metrics in self.execution_metrics.values()
        )

        total_duration = sum(
            metrics["total_duration_ms"] for metrics in self.execution_metrics.values()
        )

        successful_executions = sum(
            metrics["successful_executions"] for metrics in self.execution_metrics.values()
        )

        return {
            "thread_id": self.thread_id,
            "total_executions": total_executions,
            "total_duration_ms": total_duration,
            "successful_executions": successful_executions,
            "success_rate": (
                (successful_executions / total_executions * 100) if total_executions > 0 else 0
            ),
            "avg_execution_time_ms": (
                total_duration / total_executions if total_executions > 0 else 0
            ),
            "checkpoints_created": self.checkpoints_created,
            "checkpoints_restored": self.checkpoints_restored,
            "node_metrics": self.execution_metrics,
        }


@contextmanager
def langgraph_checkpointer(config: Optional[LangGraphConfig] = None, setup_db: bool = True):
    """Context manager for synchronous LangGraph checkpointer.

    Args:
        config: LangGraph configuration
        setup_db: Whether to setup database tables

    Yields:
        Configured checkpointer instance
    """
    checkpointer = None
    try:
        checkpointer = LangGraphCheckpointerFactory.create_checkpointer(
            config=config, async_mode=False
        )

        # Setup database if needed and supported
        if setup_db and hasattr(checkpointer, "setup"):
            checkpointer.setup()

        logger.info(f"LangGraph checkpointer initialized: {type(checkpointer).__name__}")
        yield checkpointer

    except Exception as e:
        logger.error(f"Error in LangGraph checkpointer context: {e}")
        raise
    finally:
        if checkpointer and hasattr(checkpointer, "close"):
            try:
                checkpointer.close()
                logger.debug("LangGraph checkpointer closed")
            except Exception as e:
                logger.warning(f"Error closing checkpointer: {e}")


@asynccontextmanager
async def async_langgraph_checkpointer(
    config: Optional[LangGraphConfig] = None, setup_db: bool = True
):
    """Async context manager for LangGraph checkpointer.

    Args:
        config: LangGraph configuration
        setup_db: Whether to setup database tables

    Yields:
        Configured async checkpointer instance
    """
    checkpointer = None
    try:
        checkpointer = LangGraphCheckpointerFactory.create_checkpointer(
            config=config, async_mode=True
        )

        # Setup database if needed and supported
        if setup_db and hasattr(checkpointer, "asetup"):
            await checkpointer.asetup()
        elif setup_db and hasattr(checkpointer, "setup"):
            # Fallback to sync setup
            checkpointer.setup()

        logger.info(f"Async LangGraph checkpointer initialized: {type(checkpointer).__name__}")
        yield checkpointer

    except Exception as e:
        logger.error(f"Error in async LangGraph checkpointer context: {e}")
        raise
    finally:
        if checkpointer and hasattr(checkpointer, "aclose"):
            try:
                await checkpointer.aclose()
                logger.debug("Async LangGraph checkpointer closed")
            except Exception as e:
                logger.warning(f"Error closing async checkpointer: {e}")
        elif checkpointer and hasattr(checkpointer, "close"):
            try:
                checkpointer.close()
                logger.debug("LangGraph checkpointer closed")
            except Exception as e:
                logger.warning(f"Error closing checkpointer: {e}")


def create_langgraph_runtime_config(
    thread_id: str,
    user_id: Optional[str] = None,
    checkpoint_id: Optional[str] = None,
    stream_mode: str = "values",
    debug: bool = False,
    **extra_config,
) -> Dict[str, Any]:
    """Create complete runtime configuration for LangGraph.

    Args:
        thread_id: Unique thread identifier
        user_id: Optional user identifier
        checkpoint_id: Optional specific checkpoint to resume from
        stream_mode: Stream mode for graph execution
        debug: Enable debug mode
        **extra_config: Additional configuration parameters

    Returns:
        Complete runtime configuration dictionary
    """
    thread_config = LangGraphConfigManager.create_thread_config(
        thread_id=thread_id,
        user_id=user_id,
        checkpoint_id=checkpoint_id,
        **extra_config,
    )

    execution_config = LangGraphConfigManager.create_execution_config(
        stream_mode=stream_mode, debug=debug
    )

    # Merge configurations
    runtime_config = {**thread_config, **execution_config}

    logger.debug(f"Created runtime config for thread: {thread_id}")
    return runtime_config
