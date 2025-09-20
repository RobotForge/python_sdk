"""
Basic usage examples for the Telemetry SDK
Demonstrates the core integration patterns
"""

import asyncio
import time
from telemetry_sdk import TelemetryClient, quick_setup


async def example_basic_setup():
    """Example 1: Basic client setup"""
    print("🚀 Example 1: Basic Setup")
    print("-" * 40)
    
    # Option 1: Manual setup
    client = TelemetryClient(
        api_key="your-api-key",
        endpoint="https://localhost:8443",
        project_id="example-project",
        tenant_id="demo-tenant",
        user_id="demo-user",
        application_id="basic-example"
    )
    
    # Option 2: Quick setup (recommended)
    client = quick_setup(
        api_key="your-api-key",
        endpoint="https://localhost:8443",
        project_id="example-project"
    )
    
    print(f"✅ Client created for project: {client.config.project_id}")
    print(f"📡 Endpoint: {client.config.endpoint}")
    
    await client.close()


async def example_context_managers():
    """Example 2: Context managers for tracing"""
    print("\n🎯 Example 2: Context Managers")
    print("-" * 40)
    
    client = quick_setup(
        api_key="demo-key",
        endpoint="https://localhost:8443", 
        project_id="context-example"
    )
    
    # Model call tracing
    async with client.trace_model_call(
        provider="openai",
        model="gpt-4"
    ) as span:
        span.set_input("What is the meaning of life?")
        
        # Simulate LLM call
        await asyncio.sleep(0.1)
        response = "42 - The answer to the ultimate question"
        
        span.set_output(response)
        span.set_tokens(25)
        span.set_cost(0.001)
        span.set_metadata("temperature", 0.7)
    
    print("✅ Model call traced")
    
    # Tool execution tracing
    async with client.trace_tool_execution(
        tool_name="calculator"
    ) as span:
        span.set_input("2 + 2")
        span.set_metadata("operation", "addition")
        
        # Simulate tool execution
        await asyncio.sleep(0.05)
        result = 4
        
        span.set_output(str(result))
    
    print("✅ Tool execution traced")
    
    # Agent action tracing
    async with client.trace_agent_action(
        action_type="planning"
    ) as span:
        span.set_input("User wants to solve a math problem")
        
        plan = {
            "steps": [
                "Parse the mathematical expression",
                "Calculate the result", 
                "Format the response"
            ]
        }
        
        span.set_metadata("plan_steps", len(plan["steps"]))
        span.set_output("Created 3-step execution plan")
    
    print("✅ Agent action traced")
    
    await client.close()


async def example_decorators():
    """Example 3: Decorator pattern"""
    print("\n🎨 Example 3: Decorators")
    print("-" * 40)
    
    client = quick_setup(
        api_key="demo-key",
        endpoint="https://localhost:8443",
        project_id="decorator-example"
    )
    
    @client.trace_model_call_decorator(provider="openai", model="gpt-3.5-turbo")
    async def generate_story(prompt: str) -> str:
        """Generate a story using AI (automatically traced)"""
        # Simulate API call
        await asyncio.sleep(0.2)
        return f"Once upon a time, based on '{prompt}', there was a great adventure..."
    
    @client.trace_tool_execution_decorator(tool_name="web_search")
    async def search_web(query: str) -> list:
        """Search the web (automatically traced)"""
        # Simulate web search
        await asyncio.sleep(0.3)
        return [
            {"title": f"Result 1 for {query}", "url": "https://example1.com"},
            {"title": f"Result 2 for {query}", "url": "https://example2.com"}
        ]
    
    @client.trace_agent_action_decorator(action_type="research")
    async def research_topic(topic: str) -> str:
        """Research a topic (automatically traced)"""
        # Simulate research process
        search_results = await search_web(f"{topic} facts")
        story = await generate_story(f"educational content about {topic}")
        
        return f"Research complete: Found {len(search_results)} sources and generated content"
    
    # Use the decorated functions
    result = await research_topic("artificial intelligence")
    print(f"✅ Research result: {result}")
    
    await client.close()


async def example_manual_events():
    """Example 4: Manual event creation"""
    print("\n🔧 Example 4: Manual Events")
    print("-" * 40)
    
    client = quick_setup(
        api_key="demo-key",
        endpoint="https://localhost:8443",
        project_id="manual-example"
    )
    
    # Create and send individual events
    model_event = client.create_model_call_event("custom_llm")
    model_event.set_input("Hello, world!")
    model_event.set_provider("custom")
    model_event.set_model("custom-model-v1")
    model_event.start_timing()
    
    # Simulate processing
    await asyncio.sleep(0.1)
    
    model_event.end_timing()
    model_event.set_output("Hello! How can I help you today?")
    model_event.set_tokens(15)
    model_event.set_cost(0.0005)
    
    event_id = await model_event.send()
    print(f"✅ Manual model event sent: {event_id}")
    
    # Create tool execution event
    tool_event = client.create_tool_execution_event("database", "db_query")
    tool_event.set_action("select")
    tool_event.set_input("SELECT * FROM users WHERE active = true")
    tool_event.start_timing()
    
    # Simulate database query
    await asyncio.sleep(0.05)
    
    tool_event.end_timing()
    tool_event.set_output("Retrieved 1,234 active users")
    tool_event.set_metadata("rows_returned", 1234)
    
    await tool_event.send()
    print("✅ Manual tool event sent")
    
    await client.close()


async def example_batch_processing():
    """Example 5: Batch processing"""
    print("\n📦 Example 5: Batch Processing")
    print("-" * 40)
    
    client = quick_setup(
        api_key="demo-key",
        endpoint="https://localhost:8443",
        project_id="batch-example"
    )
    
    # Create a batch
    batch = client.create_batch()
    
    # Add multiple events to the batch
    for i in range(5):
        event = client.create_agent_action_event(
            f"batch_action_{i}",
            "batch_processor"
        )
        event.set_input(f"Processing item {i}")
        event.set_metadata("batch_index", i)
        event.set_metadata("batch_size", 5)
        
        # Build the event and add to batch
        built_event = event.build()
        batch.add_event(built_event, {"item_type": "test_data"})
    
    print(f"📊 Created batch with {batch.size()} events")
    print(f"📏 Total batch size: {batch.get_total_size()} bytes")
    
    # Send the batch
    response = await batch.send()
    print(f"✅ Batch sent successfully: {response.data}")
    
    await client.close()


def example_sync_context():
    """Example 6: Synchronous context managers"""
    print("\n🔄 Example 6: Sync Context Managers")
    print("-" * 40)
    
    client = quick_setup(
        api_key="demo-key",
        endpoint="https://localhost:8443",
        project_id="sync-example"
    )
    
    # Sync model call
    with client.trace_model_call_sync(
        provider="local",
        model="local-model"
    ) as span:
        span.set_input("Sync processing request")
        
        # Simulate sync processing
        time.sleep(0.1)
        
        span.set_output("Sync processing complete")
        span.set_tokens(20)
    
    print("✅ Sync model call traced")
    
    # Sync tool execution
    with client.trace_tool_execution_sync(
        tool_name="file_processor"
    ) as span:
        span.set_input("process_file.txt")
        span.set_metadata("file_size", 1024)
        
        # Simulate file processing
        time.sleep(0.05)
        
        span.set_output("File processed successfully")
    
    print("✅ Sync tool execution traced")


async def example_error_handling():
    """Example 7: Error handling and resilience"""
    print("\n⚠️ Example 7: Error Handling")
    print("-" * 40)
    
    client = quick_setup(
        api_key="demo-key",
        endpoint="https://localhost:8443",
        project_id="error-example"
    )
    
    # Example: Handling exceptions in traced functions
    async with client.trace_model_call(provider="test") as span:
        span.set_input("This might fail")
        
        try:
            # Simulate an error
            raise ValueError("Simulated API error")
        except ValueError as e:
            # Error is automatically captured by the context manager
            print(f"🚨 Caught error: {e}")
            # Span will automatically be marked as error
    
    print("✅ Error traced and handled gracefully")
    
    # Example: Manual error setting
    error_event = client.create_model_call_event("error_prone_model")
    error_event.set_input("Problematic request")
    
    try:
        # Simulate error
        raise RuntimeError("Model timeout")
    except RuntimeError as e:
        error_event.set_error(e)
    
    await error_event.send()
    print("✅ Manual error event sent")
    
    await client.close()


async def main():
    """Run all basic usage examples"""
    print("🎉 Telemetry SDK - Basic Usage Examples")
    print("=" * 50)
    
    try:
        await example_basic_setup()
        await example_context_managers()
        await example_decorators()
        await example_manual_events()
        await example_batch_processing()
        example_sync_context()
        await example_error_handling()
        
        print("\n🎊 All examples completed successfully!")
        print("\n💡 Next steps:")
        print("   - Check out advanced_agent.py for complex workflows")
        print("   - See fastapi_app.py for web framework integration")
        print("   - Try auto_instrumentation_example.py for zero-code tracing")
        
    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        print("💭 Make sure your telemetry server is running at https://localhost:8443")


if __name__ == "__main__":
    asyncio.run(main())