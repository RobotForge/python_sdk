"""
Telemetry SDK for capturing and sending telemetry events
Multi-layered approach supporting decorators, context managers, auto-instrumentation, and manual control
"""


import uuid
import time
from datetime import datetime, timezone
from typing import  Any
from client.models import EventStatus, EventType, TelemetryEvent
from client.telemetry_client import TelemetryClient
\





class EventBuilder:
    """Builder pattern for creating telemetry events"""
    
    def __init__(self, client: TelemetryClient, event_type: EventType, source_component: str):
        self.client = client
        self._event_data = {
            'event_id': f"evt_{uuid.uuid4().hex[:12]}",
            'event_type': event_type,
            'source_component': source_component,
            'session_id': client.session_id,
            'tenant_id': client.tenant_id,
            'project_id': client.project_id,
            'user_id': client.user_id,
            'application_id': client.application_id,
            'status': EventStatus.SUCCESS,
            'metadata': {}
        }
        self._start_time = None
        self._details = {}

    def set_input(self, text: str) -> 'EventBuilder':
        self._event_data['input_text'] = text
        return self

    def set_output(self, text: str) -> 'EventBuilder':
        self._event_data['output_text'] = text
        return self

    def set_tokens(self, count: int) -> 'EventBuilder':
        self._event_data['token_count'] = count
        return self

    def set_cost(self, cost: float) -> 'EventBuilder':
        self._event_data['cost'] = cost
        return self

    def set_status(self, status: EventStatus) -> 'EventBuilder':
        self._event_data['status'] = status
        return self

    def set_metadata(self, key: str, value: Any) -> 'EventBuilder':
        if self._event_data['metadata'] is None:
            self._event_data['metadata'] = {}
        self._event_data['metadata'][key] = value
        return self

    def set_details(self, **kwargs) -> 'EventBuilder':
        self._details.update(kwargs)
        return self

    def start_timing(self) -> 'EventBuilder':
        self._start_time = time.time()
        return self

    def end_timing(self) -> 'EventBuilder':
        if self._start_time:
            self._event_data['latency_ms'] = int((time.time() - self._start_time) * 1000)
        return self

    async def send(self) -> str:
        """Send the event and return event_id"""
        if not self._event_data.get('timestamp'):
            self._event_data['timestamp'] = datetime.now(timezone.utc)
        
        event = TelemetryEvent(**self._event_data)
        await self.client._send_event(event, self._details)
        return event.event_id

    def build(self) -> TelemetryEvent:
        """Build the event without sending"""
        if not self._event_data.get('timestamp'):
            self._event_data['timestamp'] = datetime.now(timezone.utc)
        return TelemetryEvent(**self._event_data)

