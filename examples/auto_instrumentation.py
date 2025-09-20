"""
Auto-Instrumentation Example
Demonstrates zero-code telemetry by automatically patching popular AI/ML libraries
"""

import asyncio
from telemetry_sdk import quick_setup, AutoInstrumentation


async def demo_zero_code_tracing():
    """Demonstrate automatic tracing with zero code changes"""
    print("🔧 Auto-Instrumentation Demo")
    print("=" * 40)
    
    # Setup telemetry client
    client = quick_setup(
        api_key="demo-key",
        endpoint="https://localhost:8443",
        project_id="auto-instrumentation-demo"
    )
    
    # Enable auto-instrumentation
    auto_instr = AutoInstrumentation(client)
    auto_instr.instrument_all()
    
    # Check what was instrumented
    status = auto_instr.get_instrumentation_status()
    print("📊 Instrumentation Status:")
    for library, instrumented in status.items():
        emoji = "✅" if instrumented else "❌"
        print(f"   {emoji} {library}")
    
    print("\n🚀 Running auto-instrumented code...")
    
    # Now any calls to instrumented libraries are automatically traced!
    
    # Example 1: OpenAI (if available)
    try:
        print("\n1. Testing OpenAI auto-instrumentation...")
        # This would be automatically traced if openai is installed
        # import openai
        # response = await openai.ChatCompletion.acreate(
        #     model="gpt-3.5-turbo",
        #     messages=[{"role": "user", "content": "Hello!"}]
        # )
        print("   📝 OpenAI calls would be automatically traced")
    except ImportError:
        print("   ℹ️ OpenAI not installed - skipping")
    
    # Example 2: Requests (HTTP calls)
    try:
        print("\n2. Testing Requests auto-instrumentation...")
        import requests
        
        # This HTTP request is automatically traced
        response = requests.get("https://httpbin.org/json", timeout=5)
        print(f"   ✅ HTTP request traced: {response.status_code}")
    except Exception as e:
        print(f"   ⚠️ Request failed (but still traced): {e}")
    
    # Example 3: LangChain (if available)
    try:
        print("\n3. Testing LangChain auto-instrumentation...")
        # from langchain.llms import OpenAI
        # llm = OpenAI()
        # result = llm("What is 2+2?")  # Automatically traced
        print("   📝 LangChain calls would be automatically traced")
    except ImportError:
        print("   ℹ️ LangChain not installed - skipping")
    
    print("\n✨ All library calls were automatically traced!")
    print("💡 No code changes needed - telemetry is captured transparently")
    
    # Cleanup
    await client.close()


def demo_selective_instrumentation():
    """Demonstrate selective instrumentation of specific libraries"""
    print("\n🎯 Selective Instrumentation Demo")
    print("=" * 40)
    
    client = quick_setup(
        api_key="demo-key",
        endpoint="https://localhost:8443",
        project_id="selective-instrumentation"
    )
    
    auto_instr = AutoInstrumentation(client)
    
    # Instrument only specific libraries
    print("📦 Instrumenting specific libraries...")
    
    # Instrument only HTTP libraries
    auto_instr.instrument_requests()
    auto_instr.instrument_httpx()
    
    # Check status
    status = auto_instr.get_instrumentation_status()
    print("📊 Selective Instrumentation Status:")
    for library, instrumented in status.items():
        emoji = "✅" if instrumented else "⚪"
        print(f"   {emoji} {library}")
    
    print("\n💡 Only HTTP libraries are instrumented")
    print("   OpenAI, LangChain calls won't be traced")
    print("   HTTP requests will be traced")


async def demo_instrumentation_with_custom_logic():
    """Demonstrate combining auto-instrumentation with manual tracing"""
    print("\n🔀 Mixed Instrumentation Demo") 
    print("=" * 40)
    
    client = quick_setup(
        api_key="demo-key",
        endpoint="https://localhost:8443",
        project_id="mixed-instrumentation"
    )
    
    # Enable auto-instrumentation
    auto_instr = AutoInstrumentation(client)
    auto_instr.instrument_all()
    
    # Manual tracing for business logic
    async with client.trace_agent_action(
        action_type="data_processing",
        source_component="custom_processor"
    ) as span:
        span.set_input("Processing user data with mixed tracing")
        
        # Step 1: Manual tool execution
        async with client.trace_tool_execution(tool_name="data_validator") as tool_span:
            tool_span.set_input("Validating input data")
            await asyncio.sleep(0.1)  # Simulate validation
            tool_span.set_output("Data validation passed")
        
        # Step 2: HTTP call (automatically traced by instrumentation)
        try:
            import requests
            response = requests.get("https://httpbin.org/uuid", timeout=5)
            span.set_metadata("external_api_status", response.status_code)
        except Exception as e:
            span.set_metadata("external_api_error", str(e))
        
        # Step 3: Manual model call
        async with client.trace_model_call(
            provider="custom",
            model="internal-model"
        ) as model_span:
            model_span.set_input("Generate summary")
            await asyncio.sleep(0.2)  # Simulate model call
            model_span.set_output("Summary generated successfully")
            model_span.set_tokens(75)
        
        span.set_output("Data processing completed with mixed tracing")
    
    print("✅ Mixed tracing completed:")
    print("   📊 Business logic: Manual tracing")
    print("   🔗 HTTP calls: Auto-instrumentation")
    print("   🧠 Model calls: Manual tracing")
    
    await client.close()


def demo_configuration_options():
    """Demonstrate auto-instrumentation configuration options"""
    print("\n⚙️ Configuration Options Demo")
    print("=" * 40)
    
    client = quick_setup(
        api_key="demo-key",
        endpoint="https://localhost:8443",
        project_id="configuration-demo"
    )
    
    auto_instr = AutoInstrumentation(client)
    
    # Get current status
    initial_status = auto_instr.get_instrumentation_status()
    print("📊 Initial status (nothing instrumented):")
    for lib, status in initial_status.items():
        print(f"   {lib}: {status}")
    
    # Instrument everything
    print("\n🔧 Instrumenting all libraries...")
    auto_instr.instrument_all()
    
    all_status = auto_instr.get_instrumentation_status()
    print("📊 After instrument_all():")
    for lib, status in all_status.items():
        emoji = "✅" if status else "❌"
        print(f"   {emoji} {lib}")
    
    # Remove all instrumentation
    print("\n🔄 Removing all instrumentation...")
    auto_instr.uninstrument_all()
    
    final_status = auto_instr.get_instrumentation_status()
    print("📊 After uninstrument_all():")
    for lib, status in final_status.items():
        emoji = "⚪" if not status else "⚠️"
        print(f"   {emoji} {lib}")
    
    print("\n💡 Instrumentation can be added/removed dynamically")


async def main():
    """Run all auto-instrumentation examples"""
    try:
        await demo_zero_code_tracing()
        demo_selective_instrumentation()
        await demo_instrumentation_with_custom_logic()
        demo_configuration_options()
        
        print("\n🎉 All auto-instrumentation examples completed!")
        print("\n💡 Key Benefits:")
        print("   ✨ Zero code changes for popular libraries")
        print("   🎯 Selective instrumentation for specific needs")
        print("   🔀 Mix with manual tracing for custom logic")
        print("   ⚙️ Dynamic enable/disable capabilities")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        print("💭 Make sure your telemetry server is running")


if __name__ == "__main__":
    asyncio.run(main())