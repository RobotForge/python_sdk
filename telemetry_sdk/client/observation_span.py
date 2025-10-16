"""
Specialized span classes for different observation types.
Following Langfuse's pattern of wrapping event builders with domain-specific interfaces.
"""

from typing import TYPE_CHECKING, Any, Dict, Optional, Union
from datetime import datetime

from .event_builder import (
    EventBuilder,
    ModelCallEventBuilder, 
    ToolExecutionEventBuilder,
    AgentActionEventBuilder
)
from .models import EventStatus

if TYPE_CHECKING:
    from .telemetry_client import TelemetryClient


class ObservationSpan:
    """Base class for observation spans that wraps EventBuilder with a cleaner interface"""
    
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
        
        # Set initial values if provided
        if input is not None:
            self._builder.set_input(self._serialize_value(input))
        if output is not None:
            self._builder.set_output(self._serialize_value(output))
        if metadata:
            for key, value in metadata.items():
                self._builder.set_metadata(key, value)
    
    def _serialize_value(self, value: Any) -> str:
        """Serialize any value to string for storage"""
        if isinstance(value, str):
            return value
        elif isinstance(value, (dict, list)):
            import json
            return json.dumps(value, default=str)
        else:
            return str(value)
    
    def update(
        self,
        *,
        input: Optional[Any] = None,
        output: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> 'ObservationSpan':
        """Update span with new data (batch update pattern like Langfuse)"""
        if input is not None:
            self._builder.set_input(self._serialize_value(input))
        if output is not None:
            self._builder.set_output(self._serialize_value(output))
        if metadata:
            for key, value in metadata.items():
                self._builder.set_metadata(key, value)
        return self
    
    def set_status(self, status: Union[EventStatus, str]) -> 'ObservationSpan':
        """Set the status of the observation"""
        if isinstance(status, str):
            status = EventStatus(status)
        self._builder.set_status(status)
        return self
    
    def set_error(self, error: Exception) -> 'ObservationSpan':
        """Set error information"""
        self._builder.set_error(error)
        return self
    
    def end(self) -> 'ObservationSpan':
        """End the span and finalize timing"""
        self._builder.end_timing()
        return self
    
    @property
    def event_id(self) -> str:
        """Get the event ID"""
        return self._builder._event.event_id
    
    @property
    def trace_id(self) -> Optional[str]:
        """Get the trace ID"""
        return self._builder._event.trace_id


class ModelCallSpan(ObservationSpan):
    """Specialized span for model/LLM calls with generation-specific methods"""
    
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
        
        # Set model-specific attributes
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
        """Update model call span with generation-specific data"""
        # Call parent update for common fields
        super().update(input=input, output=output, metadata=metadata)
        
        # Update model-specific fields
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
        """Set model parameters (temperature, max_tokens, etc.)"""
        if 'temperature' in params:
            self._model_builder.set_temperature(params['temperature'])
        # Store full parameters in metadata
        self._builder.set_metadata('model_parameters', params)
        return self
    
    def set_usage_details(self, usage: Dict[str, int]) -> 'ModelCallSpan':
        """Set token usage details"""
        total_tokens = usage.get('total_tokens') or (
            usage.get('prompt_tokens', 0) + usage.get('completion_tokens', 0)
        )
        if total_tokens:
            self._builder.set_tokens(total_tokens)
        
        # Store detailed usage in metadata
        self._builder.set_metadata('usage_details', usage)
        return self
    
    def set_cost_details(self, cost: Union[float, Dict[str, float]]) -> 'ModelCallSpan':
        """Set cost information"""
        if isinstance(cost, dict):
            total_cost = cost.get('total_cost') or sum(cost.values())
            self._builder.set_cost(total_cost)
            self._builder.set_metadata('cost_details', cost)
        else:
            self._builder.set_cost(cost)
        return self
    
    def set_finish_reason(self, reason: str) -> 'ModelCallSpan':
        """Set the finish reason from the LLM"""
        self._model_builder.set_finish_reason(reason)
        return self


class ToolExecutionSpan(ObservationSpan):
    """Specialized span for tool/API executions"""
    
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
        """Update tool execution span"""
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


class AgentActionSpan(ObservationSpan):
    """Specialized span for agent actions and reasoning"""
    
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
        """Update agent action span"""
        super().update(input=input, output=output, metadata=metadata)
        
        if agent_name:
            self._agent_builder.set_agent_name(agent_name)
        if thought_process:
            self._agent_builder.set_thought_process(thought_process)
        if selected_tool:
            self._agent_builder.set_selected_tool(selected_tool)
        
        return self