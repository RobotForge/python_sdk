

from typing import Any, Dict, Optional

from client.event_builder import TelemetryEvent
from client.telemetry_client import TelemetryClient


class BatchManager:
    """Manages batch sending of events"""
    
    def __init__(self, client: TelemetryClient):
        self.client = client
        self.events = []
        self.details = []

    def add_event(self, event: TelemetryEvent, details: Optional[Dict[str, Any]] = None):
        self.events.append(event)
        self.details.append(details or {})

    async def send(self) -> Dict[str, Any]:
        """Send all batched events"""
        if not self.events:
            return {"total_events": 0, "sent": 0}
        
        payload = {
            "events": [
                {
                    "event": event.to_dict(),
                    "details": details
                }
                for event, details in zip(self.events, self.details)
            ]
        }
        
        result = await self.client._send_batch(payload)
        self.events.clear()
        self.details.clear()
        return result