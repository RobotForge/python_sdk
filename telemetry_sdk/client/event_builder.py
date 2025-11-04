"""
Event builder for creating and configuring telemetry events.
"""

import time
import uuid
import json
from datetime import datetime, timezone
from typing import Any, Dict, Optional, TYPE_CHECKING, Self

from .models import TelemetryEvent, EventType, EventStatus, EventIngestionRequest
from ..utils.exceptions import ValidationError

if TYPE_CHECKING:
    from .telemetry_client import TelemetryClient


class EventBuilder:
    """Builder for constructing telemetry events with fluent method chaining."""

    def __init__(self, client: 'TelemetryClient', event_type: EventType, source_component: str):
        self.client = client
        self._start_time: Optional[float] = None
        self._details: Dict[str, Any] = {}

        self._event = TelemetryEvent(
            event_id=f"evt_{uuid.uuid4().hex[:12]}",
            event_type=event_type,
            source_component=source_component,
            session_id=client.config.session_id,
            tenant_id=client.config.tenant_id,
            application_id=client.config.application_id,
            status=EventStatus.SUCCESS,
            timestamp=None,
            metadata={}
        )

    # ---------- Core Setters ---------- #

    def set_event_id(self, event_id: str) -> Self:
        if not event_id:
            raise ValidationError("Event ID cannot be empty")
        self._event.event_id = event_id
        return self

    def set_provider(self, provider: str) -> Self:
        """Set the model provider (e.g., 'openai', 'anthropic')."""
        if provider:
            self._event.provider = provider
            self.set_details(provider=provider)
        return self

    def set_model(self, model: str) -> Self:
        """Set the model name (e.g., 'gpt-4', 'claude-3')."""
        if model:
            self._event.model_name = model
            self.set_details(model_name=model)
        return self

    def set_finish_reason(self, reason: str) -> Self:
        """Set LLM finish reason."""
        return self.set_details(finish_reason=reason)

    def set_input(self, text: str) -> Self:
        """Set input text for the event."""
        if text is not None:
            self._event.input_text = str(text)[:10000]
        return self

    def set_output(self, text: str) -> Self:
        """Set output text for the event."""
        if text is not None:
            self._event.output_text = str(text)[:10000]
        return self

    def set_tokens(self, count: int) -> Self:
        """Set total token count."""
        if count is not None and count >= 0:
            self._event.token_count = count
        return self

    def set_cost(self, cost: float) -> Self:
        """Set cost in USD."""
        if cost is not None and cost >= 0:
            self._event.cost = cost
        return self

    # ---------- Convenience Methods ---------- #

    def set_usage_details(self, usage: Dict[str, int]) -> Self:
        """Attach token usage details (prompt_tokens, completion_tokens, total_tokens)."""
        if not usage:
            return self

        total_tokens = usage.get("total_tokens") or (
            usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
        )
        if total_tokens:
            self.set_tokens(total_tokens)

        self.set_metadata("usage_details", usage)
        return self

    def set_model_parameters(self, params: Dict[str, Any]) -> Self:
        """Attach model configuration parameters (temperature, top_p, etc.)."""
        if params:
            self.set_metadata("model_parameters", params)
        return self

    def set_status(self, status: EventStatus | str) -> Self:
        """Set status of the event."""
        if isinstance(status, EventStatus):
            self._event.status = status
        elif isinstance(status, str):
            try:
                self._event.status = EventStatus(status)
            except ValueError:
                raise ValidationError(f"Invalid status: {status}")
        return self

    def set_metadata(self, key: str, value: Any) -> Self:
        """Attach metadata key-value pair."""
        if not key:
            raise ValidationError("Metadata key cannot be empty")
        if self._event.metadata is None:
            self._event.metadata = {}
        try:
            json.dumps(value, default=str)
            self._event.metadata[key] = value
        except Exception:
            self._event.metadata[key] = str(value)
        return self

    def add_metadata(self, metadata: Dict[str, Any]) -> Self:
        """Attach multiple metadata entries."""
        if metadata:
            for key, val in metadata.items():
                self.set_metadata(key, val)
        return self

    def set_details(self, **kwargs) -> Self:
        """Set additional details to include in ingestion payload."""
        self._details.update(kwargs)
        return self

    # ---------- Trace Info ---------- #

    def set_trace_info(
        self,
        trace_id: Optional[str] = None,
        span_id: Optional[str] = None,
        parent_span_id: Optional[str] = None,
    ) -> Self:
        """Set distributed tracing identifiers."""
        if trace_id:
            self._event.trace_id = trace_id
        if span_id:
            self._event.span_id = span_id
        if parent_span_id:
            self._event.parent_span_id = parent_span_id
        return self

    def set_parent_event(self, parent_event_id: str) -> Self:
        """Set parent event ID for hierarchical event trees."""
        if parent_event_id:
            self._event.parent_event_id = parent_event_id
        return self

    # ---------- Timing ---------- #

    def start_timing(self) -> Self:
        """Start timing latency measurement."""
        self._start_time = time.time()
        return self

    def end_timing(self) -> Self:
        """Stop timing and record latency."""
        if self._start_time is not None:
            self._event.latency_ms = int((time.time() - self._start_time) * 1000)
        return self

    def set_latency(self, latency_ms: int) -> Self:
        """Manually set latency (ms)."""
        if latency_ms is not None and latency_ms >= 0:
            self._event.latency_ms = latency_ms
        return self

    # ---------- Error / Validation ---------- #

    def set_error(self, error: Exception) -> Self:
        """Mark event as errored and record exception metadata."""
        self._event.set_error(error)
        return self

    def set_timestamp(self, timestamp: datetime) -> Self:
        """Set event timestamp."""
        if timestamp:
            self._event.timestamp = timestamp.replace(tzinfo=timezone.utc)
        return self

    def _validate_event(self) -> None:
        """Validate required fields and payload size before send."""
        required = ["event_id", "event_type", "source_component", "session_id"]
        for field in required:
            if not getattr(self._event, field):
                raise ValidationError(f"Missing required field '{field}'")

        # Check payload size
        payload = str(self._event.to_dict())
        if len(payload) > self.client.config.max_payload_size:
            raise ValidationError(
                f"Event payload ({len(payload)} bytes) exceeds limit "
                f"({self.client.config.max_payload_size} bytes)"
            )

    # ---------- Build / Send ---------- #

    def build(self) -> TelemetryEvent:
        """Finalize and return TelemetryEvent without sending."""
        if not self._event.timestamp:
            self._event.timestamp = datetime.utcnow().replace(tzinfo=timezone.utc)
        self._validate_event()
        return self._event

    async def send(self) -> str:
        """Build and send telemetry event."""
        event = self.build()
        request = EventIngestionRequest(event=event, details=self._details)
        await self.client.send_event(event, self._details)
        return event.event_id

    # ---------- Context Manager Support ---------- #

    def __enter__(self) -> Self:
        self.start_timing()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_timing()
        if exc_type:
            self.set_status(EventStatus.ERROR)
            if exc_val:
                self.set_error(exc_val)


# ---------- Specialized Builders ---------- #

class ModelCallEventBuilder(EventBuilder):
    """Builder for model/LLM call events."""
    def __init__(self, client: 'TelemetryClient', source_component: str = "model_call"):
        super().__init__(client, EventType.MODEL_CALL, source_component)

    def set_temperature(self, temperature: float) -> Self:
        return self.set_details(temperature=temperature)


class ToolExecutionEventBuilder(EventBuilder):
    """Builder for tool or API call telemetry events."""
    def __init__(self, client: 'TelemetryClient', tool_name: str, source_component: str = None):
        super().__init__(client, EventType.TOOL_EXECUTION, source_component or tool_name)
        self.set_details(tool_name=tool_name)

    def set_action(self, action: str) -> Self:
        return self.set_details(action=action)

    def set_endpoint(self, endpoint: str) -> Self:
        return self.set_details(endpoint=endpoint)

    def set_http_method(self, method: str) -> Self:
        return self.set_details(http_method=method)

    def set_http_status(self, status_code: int) -> Self:
        return self.set_details(http_status_code=status_code)


class AgentActionEventBuilder(EventBuilder):
    """Builder for agent reasoning/action events."""
    def __init__(self, client: 'TelemetryClient', action_type: str, source_component: str = "agent"):
        super().__init__(client, EventType.AGENT_ACTION, source_component)
        self.set_details(action_type=action_type)

    def set_agent_name(self, name: str) -> Self:
        return self.set_details(agent_name=name)

    def set_thought_process(self, thought: str) -> Self:
        return self.set_details(thought_process=thought)

    def set_selected_tool(self, tool: str) -> Self:
        return self.set_details(selected_tool=tool)
