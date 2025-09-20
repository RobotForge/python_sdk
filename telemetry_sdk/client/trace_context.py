"""
Context manager for tracing operations with automatic event handling
"""

import asyncio
from typing import TYPE_CHECKING, Any, Dict, Optional
from contextlib import asynccontextmanager

from .event_builder import EventBuilder, ModelCallEventBuilder, ToolExecutionEventBuilder, AgentActionEventBuilder
from .models import EventType, EventStatus
from ..utils.exceptions import TelemetrySDKError

if TYPE_CHECKING:
    from .telemetry_client import TelemetryClient


class TraceContext:
    """Context manager for tracing operations with automatic timing and error handling"""
    
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
        self._auto_send = kwargs.pop('auto_send', True)

    async def __aenter__(self) -> EventBuilder:
        """Enter the async context and start tracing"""
        # Create appropriate builder based on event type
        if self.event_type == EventType.MODEL_CALL:
            self.builder = ModelCallEventBuilder(self.client, self.source_component)
            # Set provider and model if provided
            if 'provider' in self.kwargs:
                self.builder.set_provider(self.kwargs['provider'])
            if 'model' in self.kwargs:
                self.builder.set_model(self.kwargs['model'])
            if 'temperature' in self.kwargs:
                self.builder.set_temperature(self.kwargs['temperature'])
                
        elif self.event_type == EventType.TOOL_EXECUTION:
            tool_name = self.kwargs.get('tool_name', 'unknown_tool')
            self.builder = ToolExecutionEventBuilder(self.client, tool_name, self.source_component)
            # Set tool-specific details
            if 'action' in self.kwargs:
                self.builder.set_action(self.kwargs['action'])
            if 'endpoint' in self.kwargs:
                self.builder.set_endpoint(self.kwargs['endpoint'])
            if 'http_method' in self.kwargs:
                self.builder.set_http_method(self.kwargs['http_method'])
                
        elif self.event_type == EventType.AGENT_ACTION:
            action_type = self.kwargs.get('action_type', 'unknown_action')
            self.builder = AgentActionEventBuilder(self.client, action_type, self.source_component)
            # Set agent-specific details
            if 'agent_name' in self.kwargs:
                self.builder.set_agent_name(self.kwargs['agent_name'])
            if 'thought_process' in self.kwargs:
                self.builder.set_thought_process(self.kwargs['thought_process'])
                
        else:
            # Default to generic EventBuilder
            self.builder = EventBuilder(self.client, self.event_type, self.source_component)

        # Set common attributes from kwargs
        for key, value in self.kwargs.items():
            if key in ['input_text', 'output_text', 'token_count', 'cost', 'latency_ms']:
                getattr(self.builder, f'set_{key}')(value)
            elif key.startswith('meta_'):
                # Handle metadata with meta_ prefix
                self.builder.set_metadata(key[5:], value)
            elif key not in ['provider', 'model', 'temperature', 'tool_name', 'action', 
                           'endpoint', 'http_method', 'action_type', 'agent_name', 
                           'thought_process', 'auto_send']:
                # Set other attributes as metadata
                self.builder.set_metadata(key, value)

        # Start timing
        self.builder.start_timing()
        
        return self.builder

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
        
        # Send the event if auto_send is enabled
        if self._auto_send:
            try:
                await self.builder.send()
            except Exception as e:
                # Don't let telemetry errors break the application
                # Log the error if logging is available
                if hasattr(self.client, '_logger'):
                    self.client._logger.error(f"Failed to send telemetry event: {e}")


class SyncTraceContext:
    """Synchronous context manager for tracing operations"""
    
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
        self._auto_send = kwargs.pop('auto_send', True)

    def __enter__(self) -> EventBuilder:
        """Enter the sync context and start tracing"""
        # Create appropriate builder (same logic as async version)
        if self.event_type == EventType.MODEL_CALL:
            self.builder = ModelCallEventBuilder(self.client, self.source_component)
        elif self.event_type == EventType.TOOL_EXECUTION:
            tool_name = self.kwargs.get('tool_name', 'unknown_tool')
            self.builder = ToolExecutionEventBuilder(self.client, tool_name, self.source_component)
        elif self.event_type == EventType.AGENT_ACTION:
            action_type = self.kwargs.get('action_type', 'unknown_action')
            self.builder = AgentActionEventBuilder(self.client, action_type, self.source_component)
        else:
            self.builder = EventBuilder(self.client, self.event_type, self.source_component)

        # Set attributes from kwargs (same logic as async version)
        for key, value in self.kwargs.items():
            if key in ['input_text', 'output_text', 'token_count', 'cost', 'latency_ms']:
                getattr(self.builder, f'set_{key}')(value)
            elif key.startswith('meta_'):
                self.builder.set_metadata(key[5:], value)
            elif key not in ['provider', 'model', 'temperature', 'tool_name', 'action', 
                           'endpoint', 'http_method', 'action_type', 'agent_name', 
                           'thought_process', 'auto_send']:
                self.builder.set_metadata(key, value)

        # Start timing
        self.builder.start_timing()
        
        return self.builder

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
                # Build the event and queue it
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