"""
Core data models for the Telemetry SDK
"""

from datetime import datetime
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict, field
from enum import Enum
import uuid


class EventType(str, Enum):
    """Event type enumeration"""
    MODEL_CALL = "model_call"
    TOOL_EXECUTION = "tool_execution"
    MCP_EVENT = "mcp_event"
    AGENT_ACTION = "agent_action"


class EventStatus(str, Enum):
    """Event status enumeration"""
    SUCCESS = "success"
    ERROR = "error"
    PENDING = "pending"


@dataclass
class TelemetryEvent:
    """Core telemetry event structure"""
    event_id: str
    event_type: EventType
    session_id: str
    # tenant_id: str
    # project_id: str
    # user_id: str
    application_id: str
    source_component: str
    operation_name: Optional[str] = None
    service_name: Optional[str] = None
    tags: Optional[Dict[str, str]] = field(default_factory=dict)
    provider: Optional[str] = None           # LLM provider
    model_name: Optional[str] = None         # Model identifier
    top_p: Optional[float] = None  
    status: EventStatus = EventStatus.SUCCESS
    timestamp: Optional[datetime] = None
    parent_event_id: Optional[str] = None
    input_text: Optional[str] = None
    output_text: Optional[str] = None
    token_count: Optional[int] = None
    latency_ms: Optional[int] = None
    cost: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    parent_span_id: Optional[str] = None

    def __post_init__(self):
        """Post-initialization validation and defaults"""
        if not self.event_id:
            self.event_id = f"evt_{uuid.uuid4().hex[:12]}"
        
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
        
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for API payload"""
        data = asdict(self)
        if self.timestamp:
            data['timestamp'] = self.timestamp.isoformat()
        return data

    def add_metadata(self, key: str, value: Any) -> 'TelemetryEvent':
        """Add metadata to the event"""
        if self.metadata is None:
            self.metadata = {}
        self.metadata[key] = value
        return self

    def set_error(self, error: Exception) -> 'TelemetryEvent':
        """Set event as error with exception details"""
        self.status = EventStatus.ERROR
        self.add_metadata("error_type", type(error).__name__)
        self.add_metadata("error_message", str(error))
        return self
    def add_tag(self, key: str, value: str) -> 'TelemetryEvent':
       if self.tags is None:
           self.tags = {}
       self.tags[key] = value
       return self
   
    def set_operation(self, operation_name: str, service_name: str = None) -> 'TelemetryEvent':
        self.operation_name = operation_name
        if service_name:
            self.service_name = service_name
        return self


@dataclass
class EventIngestionRequest:
    """Request model for event ingestion"""
    event: TelemetryEvent
    details: Optional[Dict[str, Any]] = field(default_factory=dict)
    

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API request"""
        return {
            "event": self.event.to_dict(),
            "details": self.details or {}
        }


@dataclass
class BatchEventIngestionRequest:
    """Request model for batch event ingestion"""
    events: list[EventIngestionRequest]
    batch_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API request"""
        return {
            "events": [event_req.to_dict() for event_req in self.events]
        }


@dataclass
class TelemetryConfig:
    """Configuration for telemetry client"""
    api_key: str
    endpoint: str = "https://cloud.robotforge.com.ng"
    # project_id: str
    # tenant_id: str = "default"
    # user_id: str = "default"
    application_id: str = "default"
    session_id: Optional[str] = None
    auto_send: bool = True
    batch_size: int = 50
    batch_timeout: float = 5.0
    pii_scrubbing: bool = False
    max_payload_size: int = 100_000
    retry_attempts: int = 3
    request_timeout: float = 30.0
    
    def __post_init__(self):
        """Post-initialization validation"""
        if not self.api_key:
            raise ValueError("api_key is required")
        # if not self.endpoint:
        #     raise ValueError("endpoint is required")
        # if not self.project_id:
        #     raise ValueError("project_id is required")
        if self.endpoint is not "https://cloud.robotforge.com.ng":
            raise ValueError("Invalid endpoint. We do not support external endpoints yet")
        
        self.endpoint = self.endpoint.rstrip('/')
        
        if self.session_id is None:
            self.session_id = f"sess_{uuid.uuid4().hex[:12]}"


@dataclass
class APIResponse:
    """Standard API response wrapper"""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    status_code: Optional[int] = None

    @classmethod
    def success_response(cls, data: Dict[str, Any], status_code: int = 200) -> 'APIResponse':
        """Create a success response"""
        return cls(success=True, data=data, status_code=status_code)

    @classmethod
    def error_response(cls, error: str, status_code: int = 500) -> 'APIResponse':
        """Create an error response"""
        return cls(success=False, error=error, status_code=status_code)