"""
Client module for the Telemetry SDK
"""

from .telemetry_client import TelemetryClient
from .event_builder import (
    EventBuilder, 
    ModelCallEventBuilder, 
    ToolExecutionEventBuilder, 
    AgentActionEventBuilder
)
from .trace_context import TraceContext, SyncTraceContext, trace_operation, trace_sync_operation
from .batch_manager import BatchManager, AutoBatchManager, BatchStats
from .models import (
    TelemetryEvent, 
    EventType, 
    EventStatus, 
    TelemetryConfig,
    EventIngestionRequest,
    BatchEventIngestionRequest,
    APIResponse
)

# ✅ NEW IMPORTS: Specialized span classes
from .observation_span import (
    ObservationSpan,
    ModelCallSpan,
    ToolExecutionSpan,
    AgentActionSpan
)

# ✅ NEW IMPORT: LLM pricing database
from .pricing import LLMPricing


__all__ = [
    # Main client
    "TelemetryClient",
    
    # Event builders
    "EventBuilder",
    "ModelCallEventBuilder", 
    "ToolExecutionEventBuilder",
    "AgentActionEventBuilder",

    
    # ✅ NEW: Specialized span classes
    "ObservationSpan",
    "ModelCallSpan",
    "ToolExecutionSpan",
    "AgentActionSpan",
    
    # ✅ NEW: LLM pricing
    "LLMPricing",
    
    # Context managers
    "TraceContext",
    "SyncTraceContext", 
    "trace_operation",
    "trace_sync_operation",
    
    # Batch management
    "BatchManager",
    "AutoBatchManager",
    "BatchStats",
    
    # Models
    "TelemetryEvent",
    "EventType",
    "EventStatus", 
    "TelemetryConfig",
    "EventIngestionRequest",
    "BatchEventIngestionRequest",
    "APIResponse",
]