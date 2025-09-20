# ==============================================================================
# EXAMPLES __init__.py FILE  
# ==============================================================================

# Save this as: telemetry_project/examples/__init__.py
examples_init_content = """
Telemetry SDK Examples Package
Contains comprehensive examples demonstrating SDK usage patterns
"""

import sys
from pathlib import Path

# Add project root to path for clean imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import and verify SDK availability
try:
    from telemetry_sdk import TelemetryClient, quick_setup
    SDK_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Telemetry SDK not available: {e}")
    print("💡 Make sure you have the required dependencies:")
    print("   pip install aiohttp pydantic")
    SDK_AVAILABLE = False

def check_environment():
    """Check if the environment is properly set up for examples"""
    if not SDK_AVAILABLE:
        return False, "Telemetry SDK not available"
    
    # Check required dependencies
    missing_deps = []
    for dep in ['aiohttp', 'pydantic']:
        try:
            __import__(dep)
        except ImportError:
            missing_deps.append(dep)
    
    if missing_deps:
        return False, f"Missing dependencies: {', '.join(missing_deps)}"
    
    return True, "Environment ready"

def setup_example_environment():
    """Setup environment for running examples"""
    print("🔧 Setting up example environment...")
    
    ready, message = check_environment()
    if ready:
        print(f"✅ {message}")
        print(f"📍 Project root: {project_root}")
        print(f"📦 SDK location: {Path(__import__('telemetry_sdk').__file__).parent}")
        return True
    else:
        print(f"❌ {message}")
        return False

# List available examples
AVAILABLE_EXAMPLES = {
    "basic": "Basic SDK usage patterns (context managers, decorators, events)",
    "logging": "Logging integration and structured telemetry",
    "agent": "Advanced AI agent with comprehensive telemetry",
    "auto_instrumentation": "Zero-code automatic library instrumentation",
    "fastapi": "Web framework integration example",
}

def list_examples():
    """List all available examples"""
    print("📚 Available Examples:")
    for name, description in AVAILABLE_EXAMPLES.items():
        print(f"   📄 {name}_example.py - {description}")

def run_example(example_name: str):
    """Run a specific example by name"""
    if not SDK_AVAILABLE:
        print("❌ Cannot run examples: SDK not available")
        return False
    
    example_file = project_root / "examples" / f"{example_name}_example.py"
    if not example_file.exists():
        print(f"❌ Example not found: {example_file}")
        list_examples()
        return False
    
    print(f"🚀 Running {example_name} example...")
    # Import and run the example
    import importlib.util
    spec = importlib.util.spec_from_file_location(f"{example_name}_example", example_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # Run main if available
    if hasattr(module, 'main'):
        import asyncio
        if asyncio.iscoroutinefunction(module.main):
            asyncio.run(module.main())
        else:
            module.main()
    return True

if __name__ == "__main__":
    setup_example_environment()
    list_examples()
