"""
Enhanced context managers that return specialized span objects.
Improved for cleaner integration with TelemetryClient.send_event().
"""

import asyncio
from typing import TYPE_CHECKING, Any, Dict, Optional, Union
from contextlib import asynccontextmanager

from .event_builder import EventBuilder, ModelCallEventBuilder, ToolExecutionEventBuilder, AgentActionEventBuilder
from .observation_span import ObservationSpan, ModelCallSpan, ToolExecutionSpan, AgentActionSpan
from .models import EventType, EventStatus, EventIngestionRequest

if TYPE_CHECKING:
    from .telemetry_client import TelemetryClient

"""Enter sync trace context"""
from .observation_span import (
    ModelCallSpan,
    ToolExecutionSpan,
    AgentActionSpan,
    ObservationSpan,
)


class TraceContext:
    """Async context manager that returns specialized span objects for telemetry."""

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
        self._params = dict(kwargs)

        self.builder: Optional[EventBuilder] = None
        self.span: Optional[ObservationSpan] = None
        self._auto_send = self._params.pop('auto_send', True)

        self._input = self._params.pop('input', None)
        self._output = self._params.pop('output', None)
        self._metadata = self._params.pop('metadata', None)

    async def __aenter__(self) -> Union[ModelCallSpan, ToolExecutionSpan, AgentActionSpan, ObservationSpan]:
        """Enter async context and return appropriate span object."""

        if self.event_type == EventType.MODEL_CALL:
            self.builder = ModelCallEventBuilder(self.client, self.source_component)
            provider = self._params.pop('provider', None)
            model = self._params.pop('model', None)
            model_parameters = self._params.pop('model_parameters', None)
            usage_details = self._params.pop('usage_details', None)
            cost_details = self._params.pop('cost_details', None)

            if provider:
                try:
                    self.builder.set_provider(provider)
                except Exception:
                    if hasattr(self.builder, "_event"):
                        self.builder._event.provider = provider

            if model:
                try:
                    self.builder.set_model(model)
                except Exception:
                    if hasattr(self.builder, "_event"):
                        self.builder._event.model_name = model

            if 'temperature' in self._params:
                temp = self._params.pop('temperature')
                if hasattr(self.builder, "set_temperature"):
                    self.builder.set_temperature(temp)

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
                cost_details=cost_details,
            )

        elif self.event_type == EventType.TOOL_EXECUTION:
            tool_name = self._params.pop('tool_name', 'unknown_tool')
            self.builder = ToolExecutionEventBuilder(self.client, tool_name, self.source_component)

            if 'action' in self._params:
                self.builder.set_action(self._params.pop('action'))
            if 'endpoint' in self._params:
                self.builder.set_endpoint(self._params.pop('endpoint'))
            if 'http_method' in self._params:
                self.builder.set_http_method(self._params.pop('http_method'))

            self.span = ToolExecutionSpan(
                self.builder,
                self.client,
                input=self._input,
                output=self._output,
                metadata=self._metadata,
                tool_name=tool_name,
            )

        elif self.event_type == EventType.AGENT_ACTION:
            action_type = self._params.pop('action_type', 'unknown_action')
            self.builder = AgentActionEventBuilder(self.client, action_type, self.source_component)

            if 'agent_name' in self._params:
                self.builder.set_agent_name(self._params.pop('agent_name'))
            if 'thought_process' in self._params:
                self.builder.set_thought_process(self._params.pop('thought_process'))

            self.span = AgentActionSpan(
                self.builder,
                self.client,
                input=self._input,
                output=self._output,
                metadata=self._metadata,
                action_type=action_type,
            )

        else:
            self.builder = EventBuilder(self.client, self.event_type, self.source_component)
            self.span = ObservationSpan(
                self.builder,
                self.client,
                input=self._input,
                output=self._output,
                metadata=self._metadata,
            )

        self.builder.start_timing()
        return self.span

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context and finalize telemetry event."""
        if not self.builder:
            return

        try:
            self.builder.end_timing()
        except Exception:
            pass

        if exc_type:
            try:
                self.builder.set_status(EventStatus.ERROR)
                if exc_val:
                    self.builder.set_error(exc_val)
            except Exception:
                pass

        if self._auto_send:
            try:
                event = self.builder.build()
                details = getattr(self.builder, "_details", {}) or {}
                await self.client.send_event(event, details, immediate=True)
            except Exception as e:
                if hasattr(self.client, "_logger"):
                    try:
                        self.client._logger.error(f"[Telemetry] Failed to send telemetry event: {e}")
                    except Exception:
                        pass


class SyncTraceContext:
    """Synchronous version of TraceContext."""

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
        self._params = dict(kwargs)
        self.kwargs = kwargs  # ✅ 

        self.builder: Optional[EventBuilder] = None
        self.span: Optional[ObservationSpan] = None
        self._auto_send = self._params.pop('auto_send', True)

        self._input = self._params.pop('input', None)
        self._output = self._params.pop('output', None)
        self._metadata = self._params.pop('metadata', None)

    def __enter__(self):


        # Pick appropriate builder + wrapper based on event type
        if self.event_type == EventType.MODEL_CALL:
            self.builder = ModelCallEventBuilder(self.client, self.source_component)
            self.span = ModelCallSpan(self.builder, self.client)
        elif self.event_type == EventType.TOOL_EXECUTION:
            tool_name = self.kwargs.get("tool_name", self.source_component)
            self.builder = ToolExecutionEventBuilder(self.client, tool_name, self.source_component)
            self.span = ToolExecutionSpan(self.builder, self.client)
        elif self.event_type == EventType.AGENT_ACTION:
            self.builder = AgentActionEventBuilder(self.client, self.source_component)
            self.span = AgentActionSpan(self.builder, self.client)
        else:
            self.builder = EventBuilder(self.client, self.event_type, self.source_component)
            self.span = ObservationSpan(self.builder, self.client)

        # Start timing
        self.builder.start_timing()
        return self.span



    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit sync context and enqueue event for async sending."""
        if not self.builder:
            return

        try:
            self.builder.end_timing()
        except Exception:
            pass

        if exc_type:
            try:
                self.builder.set_status(EventStatus.ERROR)
                if exc_val:
                    self.builder.set_error(exc_val)
            except Exception:
                pass

        if self._auto_send:
            try:
                event = self.builder.build()
                details = getattr(self.builder, "_details", {}) or {}
                if hasattr(self.client, "_queue_sync_event"):
                    self.client._queue_sync_event(event, details)
            except Exception as e:
                if hasattr(self.client, "_logger"):
                    try:
                        self.client._logger.error(f"[Telemetry] Failed to queue telemetry event: {e}")
                    except Exception:
                        pass


@asynccontextmanager
async def trace_operation(client: 'TelemetryClient', event_type: EventType, source_component: str, **kwargs):
    """Factory for async tracing operations."""
    async with TraceContext(client, event_type, source_component, **kwargs) as builder:
        yield builder


def trace_sync_operation(client: 'TelemetryClient', event_type: EventType, source_component: str, **kwargs):
    """Factory for sync tracing operations."""
    return SyncTraceContext(client, event_type, source_component, **kwargs)
