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
        api_key="change-me",
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
        api_key="change-me",
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
        api_key="change-me",
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
            api_key="change-me",
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
            
            # Step 3: Response generation (logged as model call)
            start_time = time.time()
            response = await self._generate_response(result, user_input, request_id)
            response_time = int((time.time() - start_time) * 1000)
            
            self.logger.model_call(
                "Response generated",
                provider="openai",
                model="gpt-4",
                input_text=f"Context: {str(result)[:100]}",
                output_text=response[:100],
                latency_ms=response_time,
                token_count=150,
                cost=0.003,
                meta_request_id=request_id,
                meta_response_type="contextual"
            )
            
            # Step 4: Post-processing (logged as agent action)
            start_time = time.time()
            final_result = await self._post_process(response, request_id)
            post_process_time = int((time.time() - start_time) * 1000)
            
            self.logger.agent_action(
                "Post-processing completed",
                action_type="post_processing",
                agent_name="response_enhancer",
                thought_process="Added formatting and validation",
                latency_ms=post_process_time,
                meta_request_id=request_id,
                meta_enhancements_applied=["formatting", "validation", "safety_check"]
            )
            
            # Log successful completion
            total_time = int((time.time() - start_time) * 1000)
            self.logger.info(
                "Request completed successfully",
                extra={
                    "event_type": "request_completed",
                    "request_id": request_id,
                    "total_time_ms": total_time,
                    "intent": intent,
                    "response_length": len(final_result)
                }
            )
            
            return final_result
            
        except Exception as e:
            # Log error with context
            self.logger.error(
                f"Request processing failed: {str(e)}",
                extra={
                    "event_type": "request_failed",
                    "request_id": request_id,
                    "error_type": type(e).__name__,
                    "user_input": user_input[:100]
                }
            )
            raise
    
    async def _analyze_intent(self, user_input: str, request_id: str) -> str:
        """Simulate intent analysis"""
        #await asyncio.sleep(0.1)  # Simulate processing time
        
        # Simple intent classification
        if "weather" in user_input.lower():
            return "weather_query"
        elif "time" in user_input.lower():
            return "time_query"
        elif "help" in user_input.lower():
            return "help_request"
        else:
            return "general_query"
    
    async def _execute_action(self, intent: str, user_input: str, request_id: str) -> dict:
        """Execute the appropriate action based on intent"""
        #await asyncio.sleep(0.2)  # Simulate action execution
        
        if intent == "weather_query":
            return {
                "type": "weather",
                "location": "San Francisco",
                "temperature": "72°F",
                "conditions": "Sunny"
            }
        elif intent == "time_query":
            return {
                "type": "time",
                "current_time": "2:30 PM PST",
                "timezone": "Pacific"
            }
        elif intent == "help_request":
            return {
                "type": "help",
                "available_commands": ["weather", "time", "general questions"]
            }
        else:
            return {
                "type": "general",
                "query": user_input,
                "context": "general knowledge"
            }
    
    async def _generate_response(self, action_result: dict, user_input: str, request_id: str) -> str:
        """Generate natural language response"""
        #await asyncio.sleep(0.3)  # Simulate LLM call
        
        if action_result["type"] == "weather":
            return f"The weather in {action_result['location']} is {action_result['conditions']} with a temperature of {action_result['temperature']}."
        elif action_result["type"] == "time":
            return f"The current time is {action_result['current_time']} ({action_result['timezone']} timezone)."
        elif action_result["type"] == "help":
            commands = ", ".join(action_result["available_commands"])
            return f"I can help you with: {commands}. What would you like to know?"
        else:
            return f"I understand you're asking about: {action_result['query']}. Let me help you with that."
    
    async def _post_process(self, response: str, request_id: str) -> str:
        """Post-process the response"""
        #await asyncio.sleep(0.05)  # Simulate post-processing
        
        # Add some formatting and validation
        formatted_response = response.strip()
        if not formatted_response.endswith(('.', '!', '?')):
            formatted_response += '.'
        
        return f"🤖 {formatted_response}"


def demo_application_logging():
    """Demonstrate comprehensive application logging"""
    print("\n🏗️ Application Logging Demo")
    print("=" * 40)
    
    async def run_application_demo():
        app = TelemetryAwareApplication()
        
        # Test different types of requests
        test_requests = [
            ("What's the weather like?", "user_123"),
            ("What time is it?", "user_456"),
            ("I need help", "user_789"),
            ("Tell me about artificial intelligence", "user_101")
        ]
        
        print("🔄 Processing sample requests with full telemetry logging...")
        
        for user_input, user_id in test_requests:
            print(f"\n📝 Processing: '{user_input}'")
            try:
                result = await app.process_user_request(user_input, user_id)
                print(f"✅ Response: {result}")
            except Exception as e:
                print(f"❌ Error: {e}")
            
            # Small delay between requests
            #await asyncio.sleep(0.1)
        
        print("\n✅ Application demo completed")
        print("   📊 All operations logged with structured telemetry")
        print("   🔍 Check your telemetry dashboard for detailed traces")
    
    # Run the async demo
    asyncio.run(run_application_demo())


def demo_error_logging():
    """Demonstrate error logging and handling"""
    print("\n⚠️ Error Logging Demo")
    print("=" * 40)
    
    logger = setup_telemetry_logging(
        api_key="change-me",
        endpoint="https://localhost:8443",
        project_id="error-logging"
    )
    
    # Simulate various error scenarios
    print("🚨 Testing error logging scenarios...")
    
    # Model call error
    try:
        raise Exception("Model API rate limit exceeded")
    except Exception as e:
        logger.model_call(
            "Model call failed",
            provider="openai",
            model="gpt-4",
            input_text="Test prompt",
            latency_ms=5000,  # Long latency indicates timeout
            meta_error_type="rate_limit",
            meta_retry_count=3,
            meta_error_code="429"
        )
        logger.error(f"Model call error: {e}")
    
    # Tool execution error
    try:
        raise ConnectionError("Database connection timeout")
    except Exception as e:
        logger.tool_execution(
            "Database query failed",
            tool_name="user_database",
            action="fetch_user_profile",
            http_status_code=500,
            latency_ms=30000,  # 30 second timeout
            meta_error_type="timeout",
            meta_connection_attempts=3,
            meta_fallback_used=True
        )
        logger.error(f"Database error: {e}")
    
    # Agent action error
    try:
        raise ValueError("Invalid decision tree path")
    except Exception as e:
        logger.agent_action(
            "Agent decision failed",
            action_type="decision_making",
            agent_name="planning_agent",
            thought_process="Attempted to follow invalid decision path",
            meta_error_type="logic_error",
            meta_decision_confidence=0.15,  # Low confidence
            meta_fallback_strategy="default_response"
        )
        logger.error(f"Agent error: {e}")
    
    print("✅ Error scenarios logged")
    print("   📊 Errors captured with full context")
    print("   🔧 Actionable debugging information included")


def demo_performance_monitoring():
    """Demonstrate performance monitoring through logging"""
    print("\n⚡ Performance Monitoring Demo")
    print("=" * 40)
    
    logger = setup_telemetry_logging(
        api_key="change-me",
        endpoint="https://localhost:8443",
        project_id="performance-monitoring"
    )
    
    # Simulate various performance scenarios
    performance_scenarios = [
        {"name": "Fast Model Call", "latency": 150, "tokens": 50, "cost": 0.001},
        {"name": "Slow Model Call", "latency": 3000, "tokens": 200, "cost": 0.006},
        {"name": "Expensive Model Call", "latency": 500, "tokens": 1000, "cost": 0.02},
        {"name": "Efficient Tool Call", "latency": 100, "tokens": 0, "cost": 0},
        {"name": "Heavy Tool Call", "latency": 2000, "tokens": 0, "cost": 0},
    ]
    
    print("📈 Logging performance scenarios...")
    
    for scenario in performance_scenarios:
        if "Model" in scenario["name"]:
            logger.model_call(
                f"Performance test: {scenario['name']}",
                provider="openai",
                model="gpt-4",
                input_text="Performance test prompt",
                output_text="Test response",
                latency_ms=scenario["latency"],
                token_count=scenario["tokens"],
                cost=scenario["cost"],
                meta_performance_category=scenario["name"].split()[0].lower(),
                meta_test_scenario=True
            )
        else:
            logger.tool_execution(
                f"Performance test: {scenario['name']}",
                tool_name="performance_test_tool",
                action="execute",
                latency_ms=scenario["latency"],
                meta_performance_category=scenario["name"].split()[0].lower(),
                meta_test_scenario=True
            )
        
        print(f"   📊 {scenario['name']}: {scenario['latency']}ms")
    
    print("✅ Performance scenarios logged")
    print("   📈 Latency, cost, and efficiency metrics captured")
    print("   🎯 Performance bottlenecks easily identifiable")


async def main():
    """Run all logging integration examples"""
    print("🎉 Telemetry SDK - Logging Integration Examples")
    print("=" * 50)
    
    try:
        demo_basic_logging_integration()
        demo_advanced_logging_configuration()
        demo_structured_logging()
        demo_application_logging()
        demo_error_logging()
        demo_performance_monitoring()
        
        print("\n🎊 All logging examples completed successfully!")
        print("\n💡 Key Benefits of Logging Integration:")
        print("   ✨ Seamless integration with existing logging")
        print("   📊 Structured telemetry data automatically captured")
        print("   🔍 Rich metadata and context preservation")
        print("   ⚡ High-performance async processing")
        print("   🛡️ Error-resistant design")
        print("   🔧 Easy debugging and monitoring")
        
        print("\n🚀 Next Steps:")
        print("   - Integrate into your existing applications")
        print("   - Customize metadata for your use cases")
        print("   - Set up dashboards for your telemetry data")
        print("   - Configure alerts for error patterns")
        
    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        print("💭 Make sure your telemetry server is running")


if __name__ == "__main__":
    asyncio.run(main())