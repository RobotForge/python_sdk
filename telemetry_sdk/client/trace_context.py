"""
Enhanced context managers that return specialized span objects.
This replaces the existing trace_context.py with improved functionality.
"""

import asyncio
from typing import TYPE_CHECKING, Any, Dict, Optional, Union
from contextlib import asynccontextmanager

from .event_builder import EventBuilder, ModelCallEventBuilder, ToolExecutionEventBuilder, AgentActionEventBuilder
from .observation_span import ObservationSpan, ModelCallSpan, ToolExecutionSpan, AgentActionSpan
from .models import EventType, EventStatus

if TYPE_CHECKING:
    from .telemetry_client import TelemetryClient


class TraceContext:
    """
    Enhanced context manager that returns specialized span objects.
    Supports passing input/output directly and follows Langfuse patterns.
    """
    
    def __init__(
        self, 
        client: 'TelemetryClient', 
        event_type: EventType, 
        source_component: str,
        **kwargs
    ):
        self.client = client
        self.event_type = event_type
        self.source_component = source_component
        self.kwargs = kwargs
        self.builder: Optional[EventBuilder] = None
        self.span: Optional[ObservationSpan] = None
        self._auto_send = kwargs.pop('auto_send', True)
        
        # Extract input/output/metadata if provided
        self._input = kwargs.pop('input', None)
        self._output = kwargs.pop('output', None)
        self._metadata = kwargs.pop('metadata', None)

    async def __aenter__(self) -> Union[ModelCallSpan, ToolExecutionSpan, AgentActionSpan, ObservationSpan]:
        """Enter the async context and return appropriate span object"""
        
        # Create appropriate builder based on event type
        if self.event_type == EventType.MODEL_CALL:
            self.builder = ModelCallEventBuilder(self.client, self.source_component)
            
            # Extract model-specific parameters
            provider = self.kwargs.pop('provider', None)
            model = self.kwargs.pop('model', None)
            model_parameters = self.kwargs.pop('model_parameters', None)
            usage_details = self.kwargs.pop('usage_details', None)
            cost_details = self.kwargs.pop('cost_details', None)
            
            # Set builder attributes
            if provider:
                self.builder.set_provider(provider)
            if model:
                self.builder.set_model(model)
            if 'temperature' in self.kwargs:
                self.builder.set_temperature(self.kwargs.pop('temperature'))
            
            # Create specialized span
            self.span = ModelCallSpan(
                self.builder,
                self.client,
                input=self._input,
                output=self._output,
                metadata=self._metadata,
                provider=provider,
                model=model,
                model_parameters=model_parameters,
                usage_details=usage_details,
                cost_details=cost_details
            )
                
        elif self.event_type == EventType.TOOL_EXECUTION:
            tool_name = self.kwargs.pop('tool_name', 'unknown_tool')
            self.builder = ToolExecutionEventBuilder(self.client, tool_name, self.source_component)
            
            # Set tool-specific details
            if 'action' in self.kwargs:
                self.builder.set_action(self.kwargs.pop('action'))
            if 'endpoint' in self.kwargs:
                self.builder.set_endpoint(self.kwargs.pop('endpoint'))
            if 'http_method' in self.kwargs:
                self.builder.set_http_method(self.kwargs.pop('http_method'))
            
            # Create specialized span
            self.span = ToolExecutionSpan(
                self.builder,
                self.client,
                input=self._input,
                output=self._output,
                metadata=self._metadata,
                tool_name=tool_name
            )
                
        elif self.event_type == EventType.AGENT_ACTION:
            action_type = self.kwargs.pop('action_type', 'unknown_action')
            self.builder = AgentActionEventBuilder(self.client, action_type, self.source_component)
            
            # Set agent-specific details
            if 'agent_name' in self.kwargs:
                self.builder.set_agent_name(self.kwargs.pop('agent_name'))
            if 'thought_process' in self.kwargs:
                self.builder.set_thought_process(self.kwargs.pop('thought_process'))
            
            # Create specialized span
            self.span = AgentActionSpan(
                self.builder,
                self.client,
                input=self._input,
                output=self._output,
                metadata=self._metadata,
                action_type=action_type
            )
                
        else:
            # Default to generic EventBuilder
            self.builder = EventBuilder(self.client, self.event_type, self.source_component)
            self.span = ObservationSpan(
                self.builder,
                self.client,
                input=self._input,
                output=self._output,
                metadata=self._metadata
            )

        # Start timing
        self.builder.start_timing()
        
        return self.span

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit the async context and finalize the event"""
        if self.builder is None:
            return
        
        # End timing
        self.builder.end_timing()
        
        # Handle exceptions
        if exc_type is not None:
            self.builder.set_status(EventStatus.ERROR)
            if exc_val:
                self.builder.set_error(exc_val)
        
        # Send event if auto_send is enabled
        if self._auto_send:
            try:
                event = self.builder.build()
                await self.client.send_event(event, self.builder._details)
            except Exception as e:
                # Log error but don't propagate to avoid breaking user code
                if hasattr(self.client, '_logger'):
                    self.client._logger.error(f"Failed to send telemetry event: {e}")


class SyncTraceContext:
    """
    Synchronous version of ImprovedTraceContext.
    """
    
    def __init__(
        self, 
        client: 'TelemetryClient', 
        event_type: EventType, 
        source_component: str,
        **kwargs
    ):
        self.client = client
        self.event_type = event_type
        self.source_component = source_component
        self.kwargs = kwargs
        self.builder: Optional[EventBuilder] = None
        self.span: Optional[ObservationSpan] = None
        self._auto_send = kwargs.pop('auto_send', True)
        
        # Extract input/output/metadata if provided
        self._input = kwargs.pop('input', None)
        self._output = kwargs.pop('output', None)
        self._metadata = kwargs.pop('metadata', None)

    def __enter__(self) -> Union[ModelCallSpan, ToolExecutionSpan, AgentActionSpan, ObservationSpan]:
        """Enter the sync context and return appropriate span object"""
        
        # Create appropriate builder based on event type
        if self.event_type == EventType.MODEL_CALL:
            self.builder = ModelCallEventBuilder(self.client, self.source_component)
            
            # Extract model-specific parameters
            provider = self.kwargs.pop('provider', None)
            model = self.kwargs.pop('model', None)
            model_parameters = self.kwargs.pop('model_parameters', None)
            usage_details = self.kwargs.pop('usage_details', None)
            cost_details = self.kwargs.pop('cost_details', None)
            
            # Set builder attributes
            if provider:
                self.builder.set_provider(provider)
            if model:
                self.builder.set_model(model)
            if 'temperature' in self.kwargs:
                self.builder.set_temperature(self.kwargs.pop('temperature'))
            
            # Create specialized span
            self.span = ModelCallSpan(
                self.builder,
                self.client,
                input=self._input,
                output=self._output,
                metadata=self._metadata,
                provider=provider,
                model=model,
                model_parameters=model_parameters,
                usage_details=usage_details,
                cost_details=cost_details
            )
                
        elif self.event_type == EventType.TOOL_EXECUTION:
            tool_name = self.kwargs.pop('tool_name', 'unknown_tool')
            self.builder = ToolExecutionEventBuilder(self.client, tool_name, self.source_component)
            
            # Set tool-specific details
            if 'action' in self.kwargs:
                self.builder.set_action(self.kwargs.pop('action'))
            if 'endpoint' in self.kwargs:
                self.builder.set_endpoint(self.kwargs.pop('endpoint'))
            if 'http_method' in self.kwargs:
                self.builder.set_http_method(self.kwargs.pop('http_method'))
            
            # Create specialized span
            self.span = ToolExecutionSpan(
                self.builder,
                self.client,
                input=self._input,
                output=self._output,
                metadata=self._metadata,
                tool_name=tool_name
            )
                
        elif self.event_type == EventType.AGENT_ACTION:
            action_type = self.kwargs.pop('action_type', 'unknown_action')
            self.builder = AgentActionEventBuilder(self.client, action_type, self.source_component)
            
            # Set agent-specific details
            if 'agent_name' in self.kwargs:
                self.builder.set_agent_name(self.kwargs.pop('agent_name'))
            if 'thought_process' in self.kwargs:
                self.builder.set_thought_process(self.kwargs.pop('thought_process'))
            
            # Create specialized span
            self.span = AgentActionSpan(
                self.builder,
                self.client,
                input=self._input,
                output=self._output,
                metadata=self._metadata,
                action_type=action_type
            )
                
        else:
            # Default to generic EventBuilder
            self.builder = EventBuilder(self.client, self.event_type, self.source_component)
            self.span = ObservationSpan(
                self.builder,
                self.client,
                input=self._input,
                output=self._output,
                metadata=self._metadata
            )

        # Start timing
        self.builder.start_timing()
        
        return self.span

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the sync context and finalize the event"""
        if self.builder is None:
            return
        
        # End timing
        self.builder.end_timing()
        
        # Handle exceptions
        if exc_type is not None:
            self.builder.set_status(EventStatus.ERROR)
            if exc_val:
                self.builder.set_error(exc_val)
        
        # For sync context, we queue the event for async sending
        if self._auto_send:
            try:
                event = self.builder.build()
                # Add to client's sync queue for later async processing
                if hasattr(self.client, '_queue_sync_event'):
                    self.client._queue_sync_event(event, self.builder._details)
            except Exception as e:
                # Don't let telemetry errors break the application
                if hasattr(self.client, '_logger'):
                    self.client._logger.error(f"Failed to queue telemetry event: {e}")

@asynccontextmanager
async def trace_operation(
    client: 'TelemetryClient',
    event_type: EventType,
    source_component: str,
    **kwargs
):
    """
    Async context manager factory for tracing operations
    
    Usage:
        async with trace_operation(client, EventType.MODEL_CALL, "my_llm") as span:
            # Your traced operation here
            span.set_input("Hello")
            result = await some_operation()
            span.set_output(result)
    """
    async with TraceContext(client, event_type, source_component, **kwargs) as builder:
        yield builder


def trace_sync_operation(
    client: 'TelemetryClient',
    event_type: EventType,
    source_component: str,
    **kwargs
):
    """
    Sync context manager factory for tracing operations
    
    Usage:
        with trace_sync_operation(client, EventType.TOOL_EXECUTION, "my_tool") as span:
            # Your traced operation here
            span.set_input("input data")
            result = some_sync_operation()
            span.set_output(result)
    """
    return SyncTraceContext(client, event_type, source_component, **kwargs)