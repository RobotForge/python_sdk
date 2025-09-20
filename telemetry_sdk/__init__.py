"""
Telemetry SDK - Multi-layered Python SDK for AI/ML telemetry
Supports context managers, decorators, auto-instrumentation, and logging integration
"""

from .client import (
    TelemetryClient,
    TelemetryEvent,
    EventBuilder,
    TraceContext,
    BatchManager,
    EventType,
    EventStatus
)

from .instrumentation import (
    AutoInstrumentation,
    FrameworkIntegrations
)

from .logging_integration import (
    TelemetryHandler,
    TelemetryLogger,
    configure_telemetry_logging,
    setup_telemetry_logging
)

# Version
__version__ = "1.0.0"

# Main exports
__all__ = [
    # Core client
    "TelemetryClient",
    "TelemetryEvent", 
    "EventBuilder",
    "TraceContext",
    "BatchManager",
    "EventType",
    "EventStatus",
    
    # Auto-instrumentation
    "AutoInstrumentation",
    "FrameworkIntegrations",
    
    # Logging integration
    "TelemetryHandler",
    "TelemetryLogger",
    "configure_telemetry_logging",
    "setup_telemetry_logging",
]

# Convenience imports for quick setup
def quick_setup(
    api_key: str,
    endpoint: str,
    project_id: str,
    tenant_id: str = "default",
    user_id: str = "default",
    application_id: str = "default",
    enable_auto_instrumentation: bool = True,
    enable_logging: bool = True,
    **kwargs
) -> TelemetryClient:
    """
    Quick setup for telemetry with sensible defaults
    
    Args:
        api_key: Your telemetry service API key
        endpoint: Telemetry service endpoint URL
        project_id: Your project identifier
        tenant_id: Tenant identifier (defaults to 'default')
        user_id: User identifier (defaults to 'default')
        application_id: Application identifier (defaults to 'default')
        enable_auto_instrumentation: Enable automatic library instrumentation
        enable_logging: Enable logging integration
        **kwargs: Additional arguments passed to TelemetryClient
    
    Returns:
        Configured TelemetryClient instance
    
    Example:
        >>> import telemetry_sdk
        >>> client = telemetry_sdk.quick_setup(
        ...     api_key="your-key",
        ...     endpoint="https://telemetry.example.com",
        ...     project_id="my-project"
        ... )
        >>> 
        >>> # Use context managers
        >>> async with client.trace_model_call() as span:
        ...     # Your LLM call here
        ...     pass
        >>>
        >>> # Or use decorators
        >>> @client.trace_model_call_decorator()
        >>> async def my_function():
        ...     pass
    """
    
    # Create client
    client = TelemetryClient(
        api_key=api_key,
        endpoint=endpoint,
        project_id=project_id,
        tenant_id=tenant_id,
        user_id=user_id,
        application_id=application_id,
        **kwargs
    )
    
    # Setup auto-instrumentation if requested
    if enable_auto_instrumentation:
        auto_instr = AutoInstrumentation(client)
        auto_instr.instrument_all()
    
    # Setup logging if requested
    if enable_logging:
        configure_telemetry_logging(client)
    
    return client


# Module-level convenience functions
_default_client = None

def set_default_client(client: TelemetryClient):
    """Set the default client for module-level functions"""
    global _default_client
    _default_client = client

def get_default_client() -> TelemetryClient:
    """Get the default client"""
    if _default_client is None:
        raise RuntimeError(
            "No default telemetry client set. "
            "Call telemetry_sdk.set_default_client() or telemetry_sdk.quick_setup() first."
        )
    return _default_client

# Module-level tracing functions (use default client)
async def trace_model_call(**kwargs):
    """Module-level model call tracer using default client"""
    return get_default_client().trace_model_call(**kwargs)

async def trace_tool_execution(tool_name: str, **kwargs):
    """Module-level tool execution tracer using default client"""
    return get_default_client().trace_tool_execution(tool_name, **kwargs)

async def trace_agent_action(action_type: str, **kwargs):
    """Module-level agent action tracer using default client"""
    return get_default_client().trace_agent_action(action_type, **kwargs)

def trace_model_call_decorator(**kwargs):
    """Module-level model call decorator using default client"""
    return get_default_client().trace_model_call_decorator(**kwargs)

def trace_tool_execution_decorator(tool_name: str, **kwargs):
    """Module-level tool execution decorator using default client"""
    return get_default_client().trace_tool_execution_decorator(tool_name, **kwargs)

def trace_agent_action_decorator(action_type: str, **kwargs):
    """Module-level agent action decorator using default client"""
    return get_default_client().trace_agent_action_decorator(action_type, **kwargs)


# Add convenience functions to __all__
__all__.extend([
    "quick_setup",
    "set_default_client", 
    "get_default_client",
    "trace_model_call",
    "trace_tool_execution",
    "trace_agent_action",
    "trace_model_call_decorator",
    "trace_tool_execution_decorator", 
    "trace_agent_action_decorator"
])