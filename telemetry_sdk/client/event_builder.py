"""
Event builder for creating and configuring telemetry events
"""

import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional, TYPE_CHECKING

from .models import TelemetryEvent, EventType, EventStatus, EventIngestionRequest
from ..utils.exceptions import ValidationError

if TYPE_CHECKING:
    from .telemetry_client import TelemetryClient


class EventBuilder:
    """Builder pattern for creating telemetry events with method chaining"""
    
    def __init__(self, client: 'TelemetryClient', event_type: EventType, source_component: str):
        self.client = client
        self._start_time: Optional[float] = None
        self._details: Dict[str, Any] = {}
        
        # Initialize event with basic required fields
        self._event = TelemetryEvent(
            event_id=f"evt_{uuid.uuid4().hex[:12]}",
            event_type=event_type,
            source_component=source_component,
            session_id=client.config.session_id,
            tenant_id=client.config.tenant_id,
            # project_id=client.config.project_id,
            # user_id=client.config.user_id,
            application_id=client.config.application_id,
            status=EventStatus.SUCCESS,
            timestamp=None,  # Will be set when event is sent or built
            metadata={}
        )

    def set_event_id(self, event_id: str) -> 'EventBuilder':
        """Set custom event ID"""
        if not event_id:
            raise ValidationError("Event ID cannot be empty")
        self._event.event_id = event_id
        return self
    
    def set_provider(self, provider: str) -> 'ModelCallEventBuilder':
        """Set the model provider (e.g., 'openai', 'anthropic')"""
        self._event.provider = provider  # ✅ NEW: Set on event
        self.set_details(provider=provider)  # Keep for backwards compat
        return self

    def set_model(self, model: str) -> 'ModelCallEventBuilder':
        """Set the model name (e.g., 'gpt-4', 'claude-3')"""
        self._event.model_name = model  # ✅ NEW: Set on event
        self.set_details(model_name=model)
        return self

    def set_finish_reason(self, reason: str) -> 'ModelCallEventBuilder':
        """Set the finish reason"""
        self.set_details(finish_reason=reason)
        return self
    
    def set_input(self, text: str) -> 'EventBuilder':
        """Set input text for the event"""
        if text is not None:
            self._event.input_text = str(text)[:10000]  # Limit input size
        return self

    def set_output(self, text: str) -> 'EventBuilder':
        """Set output text for the event"""
        if text is not None:
            self._event.output_text = str(text)[:10000]  # Limit output size
        return self

    def set_tokens(self, count: int) -> 'EventBuilder':
        """Set token count for the event"""
        if count is not None and count >= 0:
            self._event.token_count = count
        return self

    def set_cost(self, cost: float) -> 'EventBuilder':
        """Set cost for the event"""
        if cost is not None and cost >= 0:
            self._event.cost = cost
        return self
    

      # ✅ NEW METHOD 1: set_usage_details()
    def set_usage_details(self, usage: Dict[str, int]) -> 'EventBuilder':
        """
        Set token usage details (prompt_tokens, completion_tokens, total_tokens).
        This is a convenience method that:
        1. Sets the token_count from total_tokens
        2. Stores full usage details in metadata
        
        Args:
            usage: Dict with keys like 'prompt_tokens', 'completion_tokens', 'total_tokens'
        
        Example:
            builder.set_usage_details({
                "prompt_tokens": 50,
                "completion_tokens": 100,
                "total_tokens": 150
            })
        """
        if usage is None:
            return self
        
        # Set token_count from total_tokens
        total_tokens = usage.get('total_tokens')
        if total_tokens is not None:
            self.set_tokens(total_tokens)
        
        # Store full usage details in metadata for detailed analysis
        self.set_metadata('usage_details', usage)
        
        return self
    
    # ✅ NEW METHOD 2: set_model_parameters()
    def set_model_parameters(self, params: Dict[str, Any]) -> 'EventBuilder':
        """
        Set model parameters (temperature, max_tokens, top_p, etc.).
        Stores parameters in metadata for analysis.
        
        Args:
            params: Dict with model configuration parameters
        
        Example:
            builder.set_model_parameters({
                "temperature": 0.7,
                "max_tokens": 500,
                "top_p": 1.0
            })
        """
        if params is None:
            return self
        
        # Store model parameters in metadata
        self.set_metadata('model_parameters', params)
        
        return self


    def set_status(self, status: EventStatus) -> 'EventBuilder':
        """Set event status"""
        if isinstance(status, EventStatus):
            self._event.status = status
        elif isinstance(status, str):
            try:
                self._event.status = EventStatus(status)
            except ValueError:
                raise ValidationError(f"Invalid status: {status}")
        return self

    def set_metadata(self, key: str, value: Any) -> 'EventBuilder':
        """Add metadata key-value pair"""
        if not key:
            raise ValidationError("Metadata key cannot be empty")
        
        if self._event.metadata is None:
            self._event.metadata = {}
        
        # Serialize complex objects to string representation
        if isinstance(value, (dict, list, tuple)):
            import json
            try:
                self._event.metadata[key] = json.dumps(value)
            except (TypeError, ValueError):
                self._event.metadata[key] = str(value)
        else:
            self._event.metadata[key] = value
        
        return self

    def add_metadata(self, metadata: Dict[str, Any]) -> 'EventBuilder':
        """Add multiple metadata entries"""
        if metadata:
            for key, value in metadata.items():
                self.set_metadata(key, value)
        return self

    def set_details(self, **kwargs) -> 'EventBuilder':
        """Set details that will be sent along with the event"""
        self._details.update(kwargs)
        return self

    def set_trace_info(
        self, 
        trace_id: Optional[str] = None, 
        span_id: Optional[str] = None, 
        parent_span_id: Optional[str] = None
    ) -> 'EventBuilder':
        """Set trace information for distributed tracing"""
        if trace_id:
            self._event.trace_id = trace_id
        if span_id:
            self._event.span_id = span_id
        if parent_span_id:
            self._event.parent_span_id = parent_span_id
        return self

    def set_parent_event(self, parent_event_id: str) -> 'EventBuilder':
        """Set parent event ID for hierarchical events"""
        if parent_event_id:
            self._event.parent_event_id = parent_event_id
        return self

    def start_timing(self) -> 'EventBuilder':
        """Start timing the event"""
        self._start_time = time.time()
        return self

    def end_timing(self) -> 'EventBuilder':
        """End timing and calculate latency"""
        if self._start_time is not None:
            latency_ms = int((time.time() - self._start_time) * 1000)
            self._event.latency_ms = latency_ms
        return self

    def set_latency(self, latency_ms: int) -> 'EventBuilder':
        """Manually set latency in milliseconds"""
        if latency_ms is not None and latency_ms >= 0:
            self._event.latency_ms = latency_ms
        return self

    def set_error(self, error: Exception) -> 'EventBuilder':
        """Set event as error with exception details"""
        self._event.set_error(error)
        return self

    def set_timestamp(self, timestamp: datetime) -> 'EventBuilder':
        """Set custom timestamp for the event"""
        if timestamp:
            self._event.timestamp = timestamp
        return self

    def build(self) -> TelemetryEvent:
        """Build the event without sending it"""
        # Set timestamp if not already set
        if self._event.timestamp is None:
            self._event.timestamp = datetime.now(timezone.utc)
        
        # Validate the event
        self._validate_event()
        
        return self._event

    async def send(self) -> str:
        """Build and send the event, returning the event ID"""
        event = self.build()
        
        # Create ingestion request
        request = EventIngestionRequest(event=event, details=self._details)
        
        # Send via client
        await self.client._send_event_request(request)
        
        return event.event_id

    def _validate_event(self) -> None:
        """Validate the event before building/sending"""
        # Check required fields
        required_fields = [
            'event_id', 'event_type', 'source_component', 
            'session_id'
        ]
        
        for field in required_fields:
            if not getattr(self._event, field):
                raise ValidationError(f"Required field '{field}' is missing")
        
        # Check payload size if configured
        if self.client.config.max_payload_size:
            payload_size = len(str(self._event.to_dict()))
            if payload_size > self.client.config.max_payload_size:
                raise ValidationError(
                    f"Event payload size ({payload_size} bytes) exceeds maximum "
                    f"allowed size ({self.client.config.max_payload_size} bytes)"
                )

    def __enter__(self) -> 'EventBuilder':
        """Support for context manager syntax"""
        self.start_timing()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """End timing on context exit"""
        self.end_timing()
        
        if exc_type is not None:
            # Set error information if exception occurred
            self.set_status(EventStatus.ERROR)
            if exc_val:
                self.set_error(exc_val)


class ModelCallEventBuilder(EventBuilder):
    """Specialized builder for model call events"""
    
    def __init__(self, client: 'TelemetryClient', source_component: str = "model_call"):
        super().__init__(client, EventType.MODEL_CALL, source_component)

    def set_provider(self, provider: str) -> 'ModelCallEventBuilder':
        """Set the model provider (e.g., 'openai', 'anthropic')"""
        return self.set_details(provider=provider)

    def set_model(self, model: str) -> 'ModelCallEventBuilder':
        """Set the model name (e.g., 'gpt-4', 'claude-3')"""
        return self.set_details(model_name=model)

    def set_temperature(self, temperature: float) -> 'ModelCallEventBuilder':
        """Set the model temperature"""
        return self.set_details(temperature=temperature)

    def set_finish_reason(self, reason: str) -> 'ModelCallEventBuilder':
        """Set the finish reason"""
        return self.set_details(finish_reason=reason)


class ToolExecutionEventBuilder(EventBuilder):
    """Specialized builder for tool execution events"""
    
    def __init__(self, client: 'TelemetryClient', tool_name: str, source_component: str = None):
        super().__init__(client, EventType.TOOL_EXECUTION, source_component or tool_name)
        self.set_details(tool_name=tool_name)

    def set_action(self, action: str) -> 'ToolExecutionEventBuilder':
        """Set the tool action"""
        return self.set_details(action=action)

    def set_endpoint(self, endpoint: str) -> 'ToolExecutionEventBuilder':
        """Set the API endpoint"""
        return self.set_details(endpoint=endpoint)

    def set_http_method(self, method: str) -> 'ToolExecutionEventBuilder':
        """Set the HTTP method"""
        return self.set_details(http_method=method)

    def set_http_status(self, status_code: int) -> 'ToolExecutionEventBuilder':
        """Set the HTTP status code"""
        return self.set_details(http_status_code=status_code)

    def set_request_payload(self, payload: Dict[str, Any]) -> 'ToolExecutionEventBuilder':
        """Set the request payload"""
        return self.set_details(request_payload=payload)

    def set_response_payload(self, payload: Dict[str, Any]) -> 'ToolExecutionEventBuilder':
        """Set the response payload"""
        return self.set_details(response_payload=payload)


class AgentActionEventBuilder(EventBuilder):
    """Specialized builder for agent action events"""
    
    def __init__(self, client: 'TelemetryClient', action_type: str, source_component: str = "agent"):
        super().__init__(client, EventType.AGENT_ACTION, source_component)
        self.set_details(action_type=action_type)

    def set_agent_name(self, name: str) -> 'AgentActionEventBuilder':
        """Set the agent name"""
        return self.set_details(agent_name=name)

    def set_thought_process(self, thought: str) -> 'AgentActionEventBuilder':
        """Set the agent's thought process"""
        return self.set_details(thought_process=thought)

    def set_selected_tool(self, tool: str) -> 'AgentActionEventBuilder':
        """Set the selected tool"""
        return self.set_details(selected_tool=tool)

    def set_target_model(self, model: str) -> 'AgentActionEventBuilder':
        """Set the target model"""
        return self.set_details(target_model=model)

    def set_policy(self, policy: Dict[str, Any]) -> 'AgentActionEventBuilder':
        """Set the agent policy"""
        return self.set_details(policy=policy)