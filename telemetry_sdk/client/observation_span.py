"""
Specialized span classes for different observation types.
Following Langfuse's pattern of wrapping event builders with domain-specific interfaces.
"""

from typing import TYPE_CHECKING, Any, Dict, Optional, Union
from datetime import datetime
import json

from .event_builder import (
    EventBuilder,
    ModelCallEventBuilder,
    ToolExecutionEventBuilder,
    AgentActionEventBuilder,
)
from .models import EventStatus

if TYPE_CHECKING:
    from .telemetry_client import TelemetryClient


class ObservationSpan:
    """Base class for observation spans that wraps EventBuilder with a cleaner interface."""

    def __init__(
        self,
        event_builder: EventBuilder,
        client: 'TelemetryClient',
        *,
        input: Optional[Any] = None,
        output: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self._builder = event_builder
        self._client = client

        # Prepopulate optional fields
        if input is not None:
            self._builder.set_input(self._serialize_value(input))
        if output is not None:
            self._builder.set_output(self._serialize_value(output))
        if metadata:
            for key, value in metadata.items():
                self._builder.set_metadata(key, value)

    def _serialize_value(self, value: Any) -> str:
        """Safely serialize arbitrary values for telemetry storage."""
        if isinstance(value, str):
            return value
        try:
            return json.dumps(value, default=str)
        except Exception:
            return str(value)

    def update(
        self,
        *,
        input: Optional[Any] = None,
        output: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> 'ObservationSpan':
        """Update span with input/output or metadata values."""
        if input is not None:
            self._builder.set_input(self._serialize_value(input))
        if output is not None:
            self._builder.set_output(self._serialize_value(output))
        if metadata:
            for key, value in metadata.items():
                self._builder.set_metadata(key, value)
        return self

    def set_status(self, status: Union[EventStatus, str]) -> 'ObservationSpan':
        """Set the current observation’s status (success, error, pending)."""
        if isinstance(status, str):
            status = EventStatus(status)
        self._builder.set_status(status)
        return self

    def set_error(self, error: Exception) -> 'ObservationSpan':
        """Mark span as errored with exception details."""
        self._builder.set_error(error)
        return self

    def end(self) -> 'ObservationSpan':
        """
        End the span and finalize timing.
        Note: does NOT send the event; sending is managed by TraceContext exit.
        """
        self._builder.end_timing()
        return self

    @property
    def event_id(self) -> str:
        """Return the unique event ID for this span."""
        return self._builder._event.event_id

    @property
    def trace_id(self) -> Optional[str]:
        """Return the trace ID for distributed tracing."""
        return self._builder._event.trace_id


# ---------------- MODEL CALL SPAN ---------------- #

class ModelCallSpan(ObservationSpan):
    """Specialized span for model/LLM calls with generation-specific metadata."""

    def __init__(
        self,
        event_builder: ModelCallEventBuilder,
        client: 'TelemetryClient',
        *,
        input: Optional[Any] = None,
        output: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        model_parameters: Optional[Dict[str, Any]] = None,
        usage_details: Optional[Dict[str, int]] = None,
        cost_details: Optional[Dict[str, float]] = None,
    ):
        super().__init__(event_builder, client, input=input, output=output, metadata=metadata)
        self._model_builder = event_builder

        if provider:
            self._model_builder.set_provider(provider)
        if model:
            self._model_builder.set_model(model)
        if model_parameters:
            self.set_model_parameters(model_parameters)
        if usage_details:
            self.set_usage_details(usage_details)
        if cost_details:
            self.set_cost_details(cost_details)

    def update(
        self,
        *,
        input: Optional[Any] = None,
        output: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        model_parameters: Optional[Dict[str, Any]] = None,
        usage_details: Optional[Dict[str, int]] = None,
        cost_details: Optional[Dict[str, float]] = None,
    ) -> 'ModelCallSpan':
        """Update span with model-specific telemetry data."""
        super().update(input=input, output=output, metadata=metadata)

        if provider:
            self._model_builder.set_provider(provider)
        if model:
            self._model_builder.set_model(model)
        if model_parameters:
            self.set_model_parameters(model_parameters)
        if usage_details:
            self.set_usage_details(usage_details)
        if cost_details:
            self.set_cost_details(cost_details)

        return self

    def set_model_parameters(self, params: Dict[str, Any]) -> 'ModelCallSpan':
        """Attach model configuration parameters like temperature, top_p, etc."""
        self._model_builder.set_model_parameters(params)
        return self

    def set_usage_details(self, usage: Dict[str, int]) -> 'ModelCallSpan':
        """Attach token usage details (prompt, completion, total)."""
        total_tokens = usage.get('total_tokens') or (
            usage.get('prompt_tokens', 0) + usage.get('completion_tokens', 0)
        )
        if total_tokens:
            self._builder.set_tokens(total_tokens)
        self._builder.set_metadata('usage_details', usage)
        return self

    def set_cost_details(self, cost: Union[float, Dict[str, float]]) -> 'ModelCallSpan':
        """Attach model call cost information."""
        if isinstance(cost, dict):
            total_cost = cost.get('total_cost') or sum(cost.values())
            self._builder.set_cost(total_cost)
            self._builder.set_metadata('cost_details', cost)
        else:
            self._builder.set_cost(cost)
        return self

    def set_finish_reason(self, reason: str) -> 'ModelCallSpan':
        """Attach LLM completion finish reason."""
        self._model_builder.set_finish_reason(reason)
        return self


# ---------------- TOOL EXECUTION SPAN ---------------- #

class ToolExecutionSpan(ObservationSpan):
    """Specialized span for tool or API executions."""

    def __init__(
        self,
        event_builder: ToolExecutionEventBuilder,
        client: 'TelemetryClient',
        *,
        input: Optional[Any] = None,
        output: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tool_name: Optional[str] = None,
    ):
        super().__init__(event_builder, client, input=input, output=output, metadata=metadata)
        self._tool_builder = event_builder

    def update(
        self,
        *,
        input: Optional[Any] = None,
        output: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
        action: Optional[str] = None,
        endpoint: Optional[str] = None,
        http_method: Optional[str] = None,
        http_status: Optional[int] = None,
    ) -> 'ToolExecutionSpan':
        """Update span with tool execution metadata."""
        super().update(input=input, output=output, metadata=metadata)

        if action:
            self._tool_builder.set_action(action)
        if endpoint:
            self._tool_builder.set_endpoint(endpoint)
        if http_method:
            self._tool_builder.set_http_method(http_method)
        if http_status:
            self._tool_builder.set_http_status(http_status)

        return self


# ---------------- AGENT ACTION SPAN ---------------- #

class AgentActionSpan(ObservationSpan):
    """Specialized span for agent reasoning and actions."""

    def __init__(
        self,
        event_builder: AgentActionEventBuilder,
        client: 'TelemetryClient',
        *,
        input: Optional[Any] = None,
        output: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
        action_type: Optional[str] = None,
    ):
        super().__init__(event_builder, client, input=input, output=output, metadata=metadata)
        self._agent_builder = event_builder

    def update(
        self,
        *,
        input: Optional[Any] = None,
        output: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
        agent_name: Optional[str] = None,
        thought_process: Optional[str] = None,
        selected_tool: Optional[str] = None,
    ) -> 'AgentActionSpan':
        """Update span with agent reasoning or selection data."""
        super().update(input=input, output=output, metadata=metadata)

        if agent_name:
            self._agent_builder.set_agent_name(agent_name)
        if thought_process:
            self._agent_builder.set_thought_process(thought_process)
        if selected_tool:
            self._agent_builder.set_selected_tool(selected_tool)

        return self
