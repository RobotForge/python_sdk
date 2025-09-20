"""
Telemetry Project - AI/ML Observability Platform
Root package for telemetry SDK, examples, and server components

Fixed version with proper imports
"""

__version__ = "1.0.0"
__author__ = "Your Organization"

# Import key components from telemetry_sdk package
# Use absolute imports to avoid issues
try:
    # Import from the telemetry_sdk package directly
    from .telemetry_sdk import (
        TelemetryClient,
        quick_setup,
        EventType,
        EventStatus,
        TelemetryConfig,
        EventBuilder,
        ModelCallEventBuilder,
        ToolExecutionEventBuilder,
        AgentActionEventBuilder
    )

    from .examples import basic_example, agent_example  
    
    # Also import auto-instrumentation if available
    try:
        from .telemetry_sdk import AutoInstrumentation, FrameworkIntegrations
        AUTO_INSTRUMENTATION_AVAILABLE = True
    except ImportError:
        AUTO_INSTRUMENTATION_AVAILABLE = False
    
    # Import logging integrations
    try:
        from .telemetry_sdk import (
            setup_telemetry_logging,
            configure_telemetry_logging,
            TelemetryLogger
        )
        LOGGING_INTEGRATION_AVAILABLE = True
    except ImportError:
        LOGGING_INTEGRATION_AVAILABLE = False
    
    SDK_AVAILABLE = True
    
    # Define what's available when importing from root
    __all__ = [
        # Core client
        "TelemetryClient",
        "quick_setup",
        "EventType", 
        "EventStatus",
        "TelemetryConfig",
        
        # Event builders
        "EventBuilder",
        "ModelCallEventBuilder",
        "ToolExecutionEventBuilder", 
        "AgentActionEventBuilder",
        
        # Utility functions
        "get_version",
        "get_project_info",
        "check_requirements"
    ]
    
    # Add auto-instrumentation to exports if available
    if AUTO_INSTRUMENTATION_AVAILABLE:
        __all__.extend(["AutoInstrumentation", "FrameworkIntegrations"])
    
    # Add logging to exports if available  
    if LOGGING_INTEGRATION_AVAILABLE:
        __all__.extend([
            "setup_telemetry_logging",
            "configure_telemetry_logging", 
            "TelemetryLogger"
        ])

except ImportError as e:
    # Handle case where SDK dependencies aren't installed
    print(f"⚠️ Telemetry SDK components not fully available: {e}")
    print("💡 Make sure you have installed the required dependencies:")
    print("   pip install aiohttp pydantic")
    
    SDK_AVAILABLE = False
    AUTO_INSTRUMENTATION_AVAILABLE = False
    LOGGING_INTEGRATION_AVAILABLE = False
    
    # Minimal exports when SDK isn't available
    __all__ = [
        "get_version",
        "get_project_info", 
        "check_requirements"
    ]


def get_version():
    """Get the version of the telemetry project"""
    return __version__


def get_project_info():
    """Get information about the project structure and available components"""
    from pathlib import Path
    
    project_root = Path(__file__).parent
    
    info = {
        "version": __version__,
        "project_root": str(project_root),
        "components": {
            "sdk": (project_root / "telemetry_sdk").exists(),
            "examples": (project_root / "examples").exists(),
            "server": (project_root / "telemetry_server").exists(),
            "tests": (project_root / "tests").exists()
        },
        "features": {
            "sdk_available": SDK_AVAILABLE,
            "auto_instrumentation": AUTO_INSTRUMENTATION_AVAILABLE,
            "logging_integration": LOGGING_INTEGRATION_AVAILABLE
        }
    }
    
    return info


def check_requirements():
    """Check if all requirements are met for the telemetry project"""
    from pathlib import Path
    
    project_root = Path(__file__).parent
    results = {
        "overall": True,
        "issues": [],
        "checks": {}
    }
    
    # Check directory structure
    required_dirs = {
        "telemetry_sdk": "Core SDK package",
        "examples": "Example code",
    }
    
    for dir_name, description in required_dirs.items():
        dir_path = project_root / dir_name
        exists = dir_path.exists()
        results["checks"][f"directory_{dir_name}"] = {
            "status": exists,
            "description": description,
            "path": str(dir_path)
        }
        
        if not exists:
            results["overall"] = False
            results["issues"].append(f"Missing directory: {dir_name} ({description})")
    
    # Check SDK availability
    results["checks"]["sdk_import"] = {
        "status": SDK_AVAILABLE,
        "description": "SDK can be imported successfully"
    }
    
    if not SDK_AVAILABLE:
        results["overall"] = False
        results["issues"].append("SDK cannot be imported - check dependencies")
    
    # Check Python dependencies
    required_deps = ["aiohttp", "pydantic"]
    for dep in required_deps:
        try:
            __import__(dep)
            status = True
        except ImportError:
            status = False
            results["overall"] = False
            results["issues"].append(f"Missing Python dependency: {dep}")
        
        results["checks"][f"dependency_{dep}"] = {
            "status": status,
            "description": f"Python package {dep}"
        }
    
    return results


def print_status():
    """Print a detailed status of the telemetry project"""
    print("🔍 Telemetry Project Status")
    print("=" * 40)
    
    info = get_project_info()
    
    print(f"📦 Version: {info['version']}")
    print(f"📁 Project root: {info['project_root']}")
    print()
    
    print("📂 Components:")
    for component, available in info["components"].items():
        status = "✅" if available else "❌"
        print(f"   {status} {component}")
    print()
    
    print("🔧 Features:")
    for feature, available in info["features"].items():
        status = "✅" if available else "❌"
        print(f"   {status} {feature}")
    print()
    
    # Run requirements check
    req_check = check_requirements()
    
    if req_check["overall"]:
        print("✅ All requirements satisfied!")
    else:
        print("❌ Issues found:")
        for issue in req_check["issues"]:
            print(f"   • {issue}")
    
    return req_check["overall"]


# Convenience function for quick testing
def test_import():
    """Test that key components can be imported and used"""
    if not SDK_AVAILABLE:
        print("❌ SDK not available - cannot run import test")
        return False
    
    try:
        print("🧪 Testing imports...")
        
        # Test basic client creation
        from telemetry_sdk import TelemetryClient, TelemetryConfig
        
        config = TelemetryConfig(
            api_key="test",
            endpoint="https://localhost:8443",
            project_id="test"
        )
        
        client = TelemetryClient(config=config)
        print("✅ TelemetryClient created successfully")
        
        # Test event builder
        from telemetry_sdk import ModelCallEventBuilder
        builder = client.create_model_call_event()
        print("✅ EventBuilder created successfully")
        
        # Test quick setup
        quick_client = quick_setup(
            api_key="test",
            endpoint="https://localhost:8443", 
            project_id="test"
        )
        print("✅ quick_setup() works")
        
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False


if __name__ == "__main__":
    # When run directly, show project status
    success = print_status()
    
    if success:
        print("\n🧪 Running import test...")
        test_import()
    
    print(f"\n💡 Usage:")
    print(f"   from telemetry_project import TelemetryClient, quick_setup")
    print(f"   from telemetry_project.telemetry_sdk import AutoInstrumentation")
    print(f"   from telemetry_project.examples import run_example")