

"""
Telemetry SDK for capturing and sending telemetry events
Multi-layered approach supporting decorators, context managers, auto-instrumentation, and manual control
"""

from client.event_builder import EventBuilder
from client.models import EventStatus, EventType
from client.telemetry_client import TelemetryClient


class TraceContext:
    """Context manager for tracing operations"""
    
    def __init__(self, client: TelemetryClient, event_type: EventType, source_component: str, **kwargs):
        self.client = client
        self.builder = EventBuilder(client, event_type, source_component)
        self.builder.start_timing()
        
        # Set initial metadata from kwargs
        for key, value in kwargs.items():
            if key in ['input_text', 'output_text', 'token_count', 'cost']:
                setattr(self.builder._event_data, key, value)
            else:
                self.builder.set_metadata(key, value)

    async def __aenter__(self) -> EventBuilder:
        return self.builder

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.builder.end_timing()
        
        if exc_type is not None:
            self.builder.set_status(EventStatus.ERROR)
            self.builder.set_metadata('error_type', exc_type.__name__)
            self.builder.set_metadata('error_message', str(exc_val))
        
        await self.builder.send()

