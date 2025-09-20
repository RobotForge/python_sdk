"""
Logging Integration Example
Demonstrates how to integrate telemetry with Python's standard logging
"""

import asyncio
import logging
import time
from telemetry_sdk import setup_telemetry_logging, configure_telemetry_logging, quick_setup


def demo_basic_logging_integration():
    """Basic telemetry logging setup and usage"""
    print("📝 Basic Logging Integration Demo")
    print("=" * 40)
    
    # Setup telemetry logging (quick method)
    logger = setup_telemetry_logging(
        api_key="demo-key",
        endpoint="https://localhost:8443",
        project_id="logging-demo",
        logger_name="my_app"
    )
    
    print("✅ Telemetry logger configured")
    
    # Use specialized telemetry logging methods
    print("\n🧠 Logging model calls...")
    logger.model_call(
        "GPT-4 response generated",
        provider="openai",
        model="gpt-4",
        input_text="What is machine learning?",
        output_text="Machine learning is a subset of AI...",
        token_count=150,
        latency_ms=250,
        cost=0.003,
        temperature=0.7,
        finish_reason="stop"
    )
    
    print("🔧 Logging tool executions...")
    logger.tool_execution(
        "Web search completed",
        tool_name="bing_search",
        action="search",
        endpoint="https://api.bing.com/search",
        http_method="GET",
        http_status_code=200,
        latency_ms=500,
        external_latency_ms=450
    )
    
    print("🤖 Logging agent actions...")
    logger.agent_action(
        "Agent completed planning phase",
        action_type="planning",
        agent_name="task_planner",
        thought_process="Analyzed user request and created 3-step execution plan",
        selected_tool="web_search",
        target_model="gpt-4"
    )
    
    print("📡 Logging MCP events...")
    logger.mcp_event(
        "MCP request processed",
        mcp_version="1.0",
        endpoint="https://mcp.example.com/api",
        request_type="get_capabilities",
        roundtrip_latency_ms=75
    )
    
    print("\n✅ All telemetry events logged successfully")


def demo_advanced_logging_configuration():
    """Advanced logging configuration with custom handlers"""
    print("\n⚙️ Advanced Logging Configuration Demo")
    print("=" * 40)
    
    # Setup telemetry client manually
    client = quick_setup(
        api_key="demo-key",
        endpoint="https://localhost:8443",
        project_id="advanced-logging"
    )
    
    # Configure telemetry logging with custom settings
    logger = configure_telemetry_logging(
        client=client,
        logger_name="advanced_app",
        level=logging.DEBUG,
        format_string="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        add_console_handler=True
    )
    
    # Add additional custom handler
    file_handler = logging.FileHandler("telemetry.log")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    logger.add_handler(file_handler)
    
    print("✅ Advanced logger configured with:")
    print("   📊 Telemetry handler (sends to server)")
    print("   🖥️ Console handler (prints to terminal)")
    print("   📁 File handler (writes to telemetry.log)")
    
    # Test different log levels
    logger.debug("Debug message with telemetry", extra={"debug_level": "verbose"})
    logger.info("Info message", extra={"info_type": "status_update"})
    logger.warning("Warning message", extra={"warning_source": "rate_limit"})
    logger.error("Error message", extra={"error_code": "E001"})
    
    print("\n✅ Multi-handler logging completed")


def demo_structured_logging():
    """Demonstrate structured logging with telemetry metadata"""
    print("\n📋 Structured Logging Demo")
    print("=" * 40)
    
    logger = setup_telemetry_logging(
        api_key="demo-key",
        endpoint="https://localhost:8443",
        project_id="structured-logging"
    )
    
    # Application context
    app_context = {
        "app_version": "1.2.3",
        "environment": "production",
        "region": "us-east-1",
        "service": "ai_assistant"
    }
    
    print("🏗️ Using structured logging with application context...")
    
    # Model call with rich context
    logger.model_call(
        "User interaction processed",
        provider="openai",
        model="gpt-4",
        input_text="Help me plan a vacation",
        output_text="I'd be happy to help you plan your vacation...",
        token_count=200,
        latency_ms=300,
        cost=0.004,
        # Additional structured data
        meta_user_id="user_12345",
        meta_session_id="session_abc789",
        meta_conversation_turn=3,
        meta_user_tier="premium",
        **{f"meta_{k}": v for k, v in app_context.items()}
    )
    
    # Tool execution with error handling
    logger.tool_execution(
        "Database query failed",
        tool_name="user_database",
        action="fetch_preferences",
        latency_ms=5000,  # Slow query
        http_status_code=500,
        # Error context
        meta_error_type="timeout",
        meta_retry_count=3,
        meta_query_complexity="high",
        **{f"meta_{k}": v for k, v in app_context.items()}
    )
    
    # Agent action with decision tracking
    logger.agent_action(
        "Routing decision made",
        action_type="routing",
        agent_name="request_router",
        thought_process="Analyzed request complexity and user tier",
        selected_tool="premium_model",
        # Decision context
        meta_decision_confidence=0.85,
        meta_alternative_tools=["basic_model", "cached_response"],
        meta_decision_factors=["user_tier", "request_complexity", "load_balancing"],
        **{f"meta_{k}": v for k, v in app_context.items()}
    )
    
    print("✅ Structured logging with rich metadata completed")


class TelemetryAwareApplication:
    """Example application class that uses telemetry logging throughout"""
    
    def __init__(self):
        self.logger = setup_telemetry_logging(
            api_key="demo-key",
            endpoint="https://localhost:8443",
            project_id="app-telemetry",
            logger_name="telemetry_app"
        )
        self.request_count = 0
    
    async def process_user_request(self, user_input: str, user_id: str = None):
        """Process user request with comprehensive logging"""
        self.request_count += 1
        request_id = f"req_{self.request_count:04d}"
        
        # Log the start of request processing
        self.logger.info(
            "Processing user request",
            extra={
                "event_type": "request_start",
                "request_id": request_id,
                "user_id": user_id,
                "input_length": len(user_input)
            }
        )
        
        try:
            # Step 1: Intent analysis (logged as model call)
            start_time = time.time()
            intent = await self._analyze_intent(user_input, request_id)
            intent_time = int((time.time() - start_time) * 1000)
            
            self.logger.model_call(
                "Intent analysis completed",
                provider="internal",
                model="intent-classifier-v2",
                input_text=user_input[:100],
                output_text=f"Intent: {intent}",
                latency_ms=intent_time,
                cost=0.001,
                meta_request_id=request_id,
                meta_confidence=0.92
            )
            
            # Step 2: Action execution (logged as tool execution)
            start_time = time.time()
            result = await self._execute_action(intent, user_input, request_id)
            action_time = int((time.time() - start_time) * 1000)
            
            self.logger.tool_execution(
                "Action executed successfully",
                tool_name=f"{intent}_handler",
                action="execute",
                latency_ms=action_time,
                meta_request_id=request_id,
                meta_result_type=type(result).__name__