"""
Telemetry SDK for capturing and sending telemetry events
Multi-layered approach supporting decorators, context managers, auto-instrumentation, and manual control
"""


from datetime import datetime
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum



class EventType(str, Enum):
    MODEL_CALL = "model_call"
    TOOL_EXECUTION = "tool_execution"
    MCP_EVENT = "mcp_event"
    AGENT_ACTION = "agent_action"


class EventStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"
    PENDING = "pending"


@dataclass
class TelemetryEvent:
    """Core telemetry event structure"""
    event_id: str
    event_type: EventType
    session_id: str
    tenant_id: str
    project_id: str
    user_id: str
    application_id: str
    source_component: str
    status: EventStatus = EventStatus.SUCCESS
    timestamp: Optional[datetime] = None
    parent_event_id: Optional[str] = None
    input_text: Optional[str] = None
    output_text: Optional[str] = None
    token_count: Optional[int] = None
    latency_ms: Optional[int] = None
    cost: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    parent_span_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for API payload"""
        data = asdict(self)
        if self.timestamp:
            data['timestamp'] = self.timestamp.isoformat()
        return data

