"""
Fixed Logging Integration Example
File: examples/logging_example.py

Fixes:
1. Auto-instrumentation patching errors
2. Event loop conflicts
3. Missing async context handling
"""

import asyncio
import logging
import time
import sys
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
    
    # Setup telemetry client manually (disable auto-instrumentation to avoid errors)
    client = quick_setup(
        api_key="demo-key",
        endpoint="https://localhost:8443",
        project_id="advanced-logging",
        enable_auto_instrumentation=False  # Disable to avoid patching errors
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
    try:
        file_handler = logging.FileHandler("telemetry.log")
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        logger.add_handler(file_handler)
        
        print("✅ Advanced logger configured with:")
        print("   📊 Telemetry handler (sends to server)")
        print("   🖥️ Console handler (prints to terminal)")
        print("   📁 File handler (writes to telemetry.log)")
    except Exception as e:
        print(f"⚠️ Could not create file handler: {e}")
        print("✅ Advanced logger configured with:")
        print("   📊 Telemetry handler (sends to server)")
        print("   🖥️ Console handler (prints to terminal)")
    
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
            )
            
            # Step 3: Response generation (logged as agent action)
            start_time = time.time()
            response = await self._generate_response(intent, result, request_id)
            response_time = int((time.time() - start_time) * 1000)
            
            self.logger.agent_action(
                "Response generated",
                action_type="response_generation",
                agent_name="response_generator",
                latency_ms=response_time,
                meta_request_id=request_id,
                meta_response_length=len(response)
            )
            
            # Log successful completion
            self.logger.info(
                "Request completed successfully",
                extra={
                    "event_type": "request_complete",
                    "request_id": request_id,
                    "total_latency_ms": intent_time + action_time + response_time,
                    "success": True
                }
            )
            
            return response
            
        except Exception as e:
            # Log error with context
            self.logger.error(
                f"Request failed: {str(e)}",
                extra={
                    "event_type": "request_error",
                    "request_id": request_id,
                    "error_type": type(e).__name__,
                    "user_input_length": len(user_input)
                }
            )
            raise
    
    async def _analyze_intent(self, user_input: str, request_id: str) -> str:
        """Simulate intent analysis"""
        await asyncio.sleep(0.1)  # Simulate processing time
        
        # Simple intent classification
        if "weather" in user_input.lower():
            return "weather_query"
        elif "help" in user_input.lower():
            return "help_request"
        elif "search" in user_input.lower():
            return "search_query"
        else:
            return "general_query"
    
    async def _execute_action(self, intent: str, user_input: str, request_id: str):
        """Simulate action execution based on intent"""
        await asyncio.sleep(0.15)  # Simulate processing time
        
        if intent == "weather_query":
            return {"weather": "sunny", "temperature": "72°F", "location": "New York"}
        elif intent == "help_request":
            return {"help_topics": ["weather", "search", "general"], "contact": "support@example.com"}
        elif intent == "search_query":
            return {"results": [{"title": "Result 1", "url": "https://example.com/1"}]}
        else:
            return {"message": "I understand you want help with something general."}
    
    async def _generate_response(self, intent: str, action_result, request_id: str) -> str:
        """Generate natural language response"""
        await asyncio.sleep(0.08)  # Simulate response generation time
        
        if intent == "weather_query":
            weather = action_result["weather"]
            temp = action_result["temperature"]
            return f"The weather is {weather} with a temperature of {temp}."
        elif intent == "help_request":
            topics = ", ".join(action_result["help_topics"])
            return f"I can help you with: {topics}. Contact {action_result['contact']} for more assistance."
        elif intent == "search_query":
            count = len(action_result["results"])
            return f"I found {count} search result(s) for your query."
        else:
            return action_result["message"]


async def demo_application_logging():
    """Demonstrate comprehensive application logging"""
    print("\n🏗️ Application Logging Demo")
    print("=" * 40)
    
    app = TelemetryAwareApplication()
    
    print("🚀 Processing sample requests with full telemetry...")
    
    # Sample requests
    requests = [
        ("What's the weather like today?", "user_123"),
        ("I need help with something", "user_456"),
        ("Search for Python tutorials", "user_789"),
        ("Hello there!", "user_000")
    ]
    
    for user_input, user_id in requests:
        try:
            print(f"\n📝 Processing: '{user_input[:30]}...'")
            response = await app.process_user_request(user_input, user_id)
            print(f"✅ Response: '{response[:50]}...'")
            
            # Small delay between requests
            await asyncio.sleep(0.1)
            
        except Exception as e:
            print(f"❌ Request failed: {e}")
    
    print(f"\n✅ Processed {len(requests)} requests with comprehensive logging")
    print("📊 Check your telemetry dashboard for detailed insights!")


async def main():
    """Run all logging integration examples"""
    print("🎉 Telemetry SDK - Logging Integration Examples")
    print("=" * 50)
    
    try:
        # Run sync demos
        demo_basic_logging_integration()
        demo_advanced_logging_configuration()
        demo_structured_logging()
        
        # Run async demo
        await demo_application_logging()
        
        print("\n🎊 All logging examples completed successfully!")
        print("\n💡 Key Features Demonstrated:")
        print("   📝 Structured logging with telemetry events")
        print("   🔧 Multiple handler configurations")
        print("   📊 Rich metadata and context tracking")
        print("   🚀 Production-ready application patterns")
        print("   🏗️ Comprehensive request lifecycle logging")
        
    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        print("💭 Note: This is a demo - telemetry server doesn't need to be running")
        print("💡 In production, events would be sent to your telemetry service")


def run_example():
    """
    Entry point that handles event loop detection
    """
    try:
        # Check if we're already in an event loop
        loop = asyncio.get_running_loop()
        print("⚠️ Running in existing event loop - creating new task")
        
        # If we're in an event loop, create a task instead of using asyncio.run()
        task = loop.create_task(main())
        return task
        
    except RuntimeError:
        # No event loop running, safe to use asyncio.run()
        return asyncio.run(main())


if __name__ == "__main__":
    # Handle both standalone execution and import from other async code
    if 'pytest' in sys.modules:
        # Running under pytest
        asyncio.run(main())
    else:
        try:
            # Try to run normally
            asyncio.run(main())
        except RuntimeError as e:
            if "cannot be called from a running event loop" in str(e):
                print("⚠️ Detected running event loop - scheduling as task")
                # Create task in existing loop
                loop = asyncio.get_running_loop()
                task = loop.create_task(main())
                print("✅ Task scheduled - will complete in background")
            else:
                raise