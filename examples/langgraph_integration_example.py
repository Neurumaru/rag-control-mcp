"""Example demonstrating LangGraph integration with MCP-RAG-Control utilities."""

import asyncio
import time
import uuid
from typing import Any, Dict

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain.chat_models import init_chat_model

# Add src to path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Import our enhanced utilities
from utils.config import get_config, set_config, AppConfig, LangGraphConfig
from utils.logger import (
    setup_logging, 
    LoggerConfig,
    log_langgraph_node_execution,
    log_langgraph_state_transition,
    create_langgraph_logger
)
from utils.langgraph_factory import (
    LangGraphCheckpointerFactory,
    LangGraphConfigManager,
    LangGraphMonitor,
    langgraph_checkpointer,
    async_langgraph_checkpointer,
    create_langgraph_runtime_config
)


class MCPRAGAgent:
    """Example MCP-RAG agent using LangGraph with enhanced utilities."""
    
    def __init__(self, config: AppConfig):
        """Initialize the agent with configuration.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.logger = create_langgraph_logger("mcp_rag_agent")
        self.model = init_chat_model(model="anthropic:claude-3-5-haiku-latest")
        self.graph = None
        self.monitor = None
        
    def build_graph(self) -> StateGraph:
        """Build the LangGraph workflow.
        
        Returns:
            Configured StateGraph instance
        """
        builder = StateGraph(MessagesState)
        
        # Add nodes
        builder.add_node("analyze_query", self._analyze_query)
        builder.add_node("retrieve_context", self._retrieve_context)
        builder.add_node("generate_response", self._generate_response)
        
        # Add edges
        builder.add_edge(START, "analyze_query")
        builder.add_edge("analyze_query", "retrieve_context")
        builder.add_edge("retrieve_context", "generate_response")
        builder.add_edge("generate_response", END)
        
        return builder
    
    def _analyze_query(self, state: MessagesState) -> Dict[str, Any]:
        """Analyze the user query to determine intent and requirements.
        
        Args:
            state: Current graph state
            
        Returns:
            Updated state with analysis results
        """
        start_time = time.time()
        thread_id = "example_thread"  # In real implementation, get from config
        
        try:
            last_message = state["messages"][-1]
            
            # Simulate query analysis
            analysis_prompt = f"Analyze this user query for intent and requirements: {last_message.content}"
            analysis_result = self.model.invoke([HumanMessage(content=analysis_prompt)])
            
            # Log successful execution
            duration_ms = (time.time() - start_time) * 1000
            log_langgraph_node_execution(
                node_name="analyze_query",
                thread_id=thread_id,
                checkpoint_id=None,
                input_data=last_message.content,
                output_data=analysis_result.content,
                duration_ms=duration_ms,
                success=True
            )
            
            # Record in monitor if available
            if self.monitor:
                self.monitor.record_node_execution("analyze_query", duration_ms, True)
            
            # Add analysis to state
            return {
                "messages": state["messages"] + [
                    AIMessage(content=f"Analysis: {analysis_result.content}", name="analyzer")
                ]
            }
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            log_langgraph_node_execution(
                node_name="analyze_query",
                thread_id=thread_id,
                checkpoint_id=None,
                input_data=last_message.content if 'last_message' in locals() else None,
                output_data=None,
                duration_ms=duration_ms,
                success=False,
                error=e
            )
            
            if self.monitor:
                self.monitor.record_node_execution("analyze_query", duration_ms, False, e)
            
            raise
    
    def _retrieve_context(self, state: MessagesState) -> Dict[str, Any]:
        """Retrieve relevant context for the query.
        
        Args:
            state: Current graph state
            
        Returns:
            Updated state with retrieved context
        """
        start_time = time.time()
        thread_id = "example_thread"
        
        try:
            # Simulate context retrieval
            analysis_message = state["messages"][-1]
            
            # In real implementation, this would query vector store, MCP tools, etc.
            context = "Retrieved context based on analysis..."
            
            duration_ms = (time.time() - start_time) * 1000
            log_langgraph_node_execution(
                node_name="retrieve_context",
                thread_id=thread_id,
                checkpoint_id=None,
                input_data=analysis_message.content,
                output_data=context,
                duration_ms=duration_ms,
                success=True
            )
            
            if self.monitor:
                self.monitor.record_node_execution("retrieve_context", duration_ms, True)
            
            # Log state transition
            log_langgraph_state_transition(
                from_node="analyze_query",
                to_node="retrieve_context",
                thread_id=thread_id,
                state_size=len(str(state))
            )
            
            return {
                "messages": state["messages"] + [
                    AIMessage(content=f"Context: {context}", name="retriever")
                ]
            }
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            log_langgraph_node_execution(
                node_name="retrieve_context",
                thread_id=thread_id,
                checkpoint_id=None,
                input_data=analysis_message.content if 'analysis_message' in locals() else None,
                output_data=None,
                duration_ms=duration_ms,
                success=False,
                error=e
            )
            
            if self.monitor:
                self.monitor.record_node_execution("retrieve_context", duration_ms, False, e)
            
            raise
    
    def _generate_response(self, state: MessagesState) -> Dict[str, Any]:
        """Generate final response using retrieved context.
        
        Args:
            state: Current graph state
            
        Returns:
            Updated state with final response
        """
        start_time = time.time()
        thread_id = "example_thread"
        
        try:
            # Get original query and context
            original_query = state["messages"][0].content
            context_message = state["messages"][-1]
            
            # Generate response
            response_prompt = f"""
            Original query: {original_query}
            Available context: {context_message.content}
            
            Generate a comprehensive response to the user's query using the provided context.
            """
            
            final_response = self.model.invoke([HumanMessage(content=response_prompt)])
            
            duration_ms = (time.time() - start_time) * 1000
            log_langgraph_node_execution(
                node_name="generate_response",
                thread_id=thread_id,
                checkpoint_id=None,
                input_data=response_prompt,
                output_data=final_response.content,
                duration_ms=duration_ms,
                success=True
            )
            
            if self.monitor:
                self.monitor.record_node_execution("generate_response", duration_ms, True)
            
            # Log state transition
            log_langgraph_state_transition(
                from_node="retrieve_context",
                to_node="generate_response",
                thread_id=thread_id,
                state_size=len(str(state))
            )
            
            return {
                "messages": state["messages"] + [final_response]
            }
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            log_langgraph_node_execution(
                node_name="generate_response",
                thread_id=thread_id,
                checkpoint_id=None,
                input_data=response_prompt if 'response_prompt' in locals() else None,
                output_data=None,
                duration_ms=duration_ms,
                success=False,
                error=e
            )
            
            if self.monitor:
                self.monitor.record_node_execution("generate_response", duration_ms, False, e)
            
            raise


def synchronous_example():
    """Example of synchronous LangGraph execution with enhanced utilities."""
    print("🚀 Starting synchronous LangGraph example...")
    
    # Setup logging
    logging_config = LoggerConfig(
        log_level="DEBUG",
        structured_logging=True,
        enable_console=True
    )
    setup_logging(logging_config)
    
    # Create configuration with LangGraph settings
    config = AppConfig()
    config.langgraph = LangGraphConfig(
        checkpointer_type="memory",
        enable_debug_mode=True,
        stream_mode="values"
    )
    set_config(config)
    
    # Initialize agent
    agent = MCPRAGAgent(config)
    
    # Create monitor
    thread_id = str(uuid.uuid4())
    agent.monitor = LangGraphMonitor(thread_id)
    
    # Build and compile graph
    builder = agent.build_graph()
    
    # Use checkpointer context manager
    with langgraph_checkpointer(config.langgraph) as checkpointer:
        graph = builder.compile(checkpointer=checkpointer)
        
        # Create runtime configuration
        runtime_config = create_langgraph_runtime_config(
            thread_id=thread_id,
            user_id="user123",
            stream_mode="values",
            debug=True
        )
        
        # Execute graph
        print("\n📝 Processing query...")
        result = graph.invoke(
            {"messages": [HumanMessage(content="What is machine learning?")]},
            config=runtime_config
        )
        
        print(f"\n✅ Response: {result['messages'][-1].content}")
        
        # Get execution summary
        summary = agent.monitor.get_execution_summary()
        print(f"\n📊 Execution Summary:")
        print(f"  - Total executions: {summary['total_executions']}")
        print(f"  - Success rate: {summary['success_rate']:.1f}%")
        print(f"  - Average execution time: {summary['avg_execution_time_ms']:.2f}ms")


async def asynchronous_example():
    """Example of asynchronous LangGraph execution with enhanced utilities."""
    print("\n🚀 Starting asynchronous LangGraph example...")
    
    # Create configuration
    config = AppConfig()
    config.langgraph = LangGraphConfig(
        checkpointer_type="memory",
        enable_debug_mode=True,
        stream_mode="values"
    )
    
    # Initialize agent
    agent = MCPRAGAgent(config)
    
    # Create monitor
    thread_id = str(uuid.uuid4())
    agent.monitor = LangGraphMonitor(thread_id)
    
    # Build graph
    builder = agent.build_graph()
    
    # Use async checkpointer context manager
    async with async_langgraph_checkpointer(config.langgraph) as checkpointer:
        graph = builder.compile(checkpointer=checkpointer)
        
        # Create runtime configuration
        runtime_config = create_langgraph_runtime_config(
            thread_id=thread_id,
            user_id="user456",
            stream_mode="values",
            debug=True
        )
        
        # Execute graph asynchronously
        print("\n📝 Processing query asynchronously...")
        result = await graph.ainvoke(
            {"messages": [HumanMessage(content="Explain neural networks")]},
            config=runtime_config
        )
        
        print(f"\n✅ Async Response: {result['messages'][-1].content}")
        
        # Stream example
        print("\n🔄 Streaming example...")
        async for chunk in graph.astream(
            {"messages": [HumanMessage(content="What is deep learning?")]},
            config=runtime_config
        ):
            if chunk.get("messages"):
                latest_message = chunk["messages"][-1]
                print(f"  Stream chunk: {latest_message.content[:50]}...")
        
        # Get execution summary
        summary = agent.monitor.get_execution_summary()
        print(f"\n📊 Async Execution Summary:")
        print(f"  - Total executions: {summary['total_executions']}")
        print(f"  - Success rate: {summary['success_rate']:.1f}%")
        print(f"  - Average execution time: {summary['avg_execution_time_ms']:.2f}ms")


def configuration_examples():
    """Demonstrate various configuration options."""
    print("\n⚙️ Configuration Examples:")
    
    # Thread configuration
    thread_config = LangGraphConfigManager.create_thread_config(
        thread_id="thread123",
        user_id="user456",
        custom_param="value"
    )
    print(f"Thread config: {thread_config}")
    
    # Execution configuration
    exec_config = LangGraphConfigManager.create_execution_config(
        stream_mode="debug",
        recursion_limit=50,
        debug=True
    )
    print(f"Execution config: {exec_config}")
    
    # Runtime configuration
    runtime_config = create_langgraph_runtime_config(
        thread_id="runtime_thread",
        user_id="runtime_user",
        stream_mode="updates",
        debug=False,
        custom_setting="enabled"
    )
    print(f"Runtime config: {runtime_config}")


if __name__ == "__main__":
    # Run configuration examples
    configuration_examples()
    
    # Run synchronous example
    synchronous_example()
    
    # Run asynchronous example
    asyncio.run(asynchronous_example())
    
    print("\n🎉 All examples completed successfully!")