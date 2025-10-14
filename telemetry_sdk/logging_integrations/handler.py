"""
Logging handler that converts log records to telemetry events
"""

import logging
import asyncio
import threading
from typing import Dict, Any, Optional, TYPE_CHECKING
from queue import Queue
from datetime import datetime, timezone

from ..client.models import EventType, EventStatus, TelemetryEvent, EventIngestionRequest

if TYPE_CHECKING:
    from ..client.telemetry_client import TelemetryClient


class TelemetryHandler(logging.Handler):
    """Logging handler that converts log records to telemetry events"""
    
    def __init__(
        self,
        client: 'TelemetryClient',
        level: int = logging.INFO,
        event_type_mapping: Optional[Dict[str, EventType]] = None,
        source_component: str = "logging"
    ):
        super().__init__(level)
        self.client = client
        self.source_component = source_component
        
        # Default mapping of log record attributes to event types
        self.event_type_mapping = event_type_mapping or {
            'model_call': EventType.MODEL_CALL,
            'tool_execution': EventType.TOOL_EXECUTION,
            'mcp_event': EventType.MCP_EVENT,
            'agent_action': EventType.AGENT_ACTION,
        }
        
        # Queue for async processing
        self._event_queue: Queue = Queue()
        self._processor_thread: Optional[threading.Thread] = None
        self._shutdown = False
        self._start_processor()

    def _start_processor(self):
        """Start background thread to process log events"""
        def processor():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._process_events())
            loop.close()
        
        self._processor_thread = threading.Thread(target=processor, daemon=True)
        self._processor_thread.start()

    async def _process_events(self):
        """Process queued events asynchronously"""
        while not self._shutdown:
            try:
                if not self._event_queue.empty():
                    event_data = self._event_queue.get(timeout=0.1)
                    request = EventIngestionRequest(
                        event=event_data['event'],
                        details=event_data['details']
                    )
                    await self.client._send_event_request(request)
                else:
                    await asyncio.sleep(0.1)
            except Exception:
                # Don't let telemetry errors break the application
                continue

    def emit(self, record: logging.LogRecord):
        """Convert log record to telemetry event"""
        try:
            # Determine event type
            event_type = self._determine_event_type(record)
            if not event_type:
                return  # Skip if can't determine event type

            # Extract telemetry data from log record
            event, details = self._extract_telemetry_data(record, event_type)
            
            # Queue for async processing
            self._event_queue.put({
                'event': event,
                'details': details
            })
            
        except Exception:
            # Never let telemetry break the application
            self.handleError(record)

    def _determine_event_type(self, record: logging.LogRecord) -> Optional[EventType]:
        """Determine event type from log record"""
        # Check if event type is explicitly specified
        if hasattr(record, 'event_type'):
            event_type_str = record.event_type
            if isinstance(event_type_str, EventType):
                return event_type_str
            elif isinstance(event_type_str, str):
                try:
                    return EventType(event_type_str)
                except ValueError:
                    pass
        
        # Check message content for keywords
        message = record.getMessage().lower()
        for key, event_type in self.event_type_mapping.items():
            if key in message:
                return event_type
        
        # Check logger name for keywords
        logger_name = record.name.lower()
        for key, event_type in self.event_type_mapping.items():
            if key in logger_name:
                return event_type
        
        # Default fallback
        return EventType.AGENT_ACTION

    def _extract_telemetry_data(self, record: logging.LogRecord, event_type: EventType) -> tuple:
        """Extract telemetry data from log record"""
        
        # Create base event
        event = TelemetryEvent(
            event_id=f"evt_log_{record.created}_{threading.get_ident()}",
            event_type=event_type,
            session_id=self.client.config.session_id,
            tenant_id=self.client.config.tenant_id,
            project_id=self.client.config.project_id,
            user_id=self.client.config.user_id,
            #application_id=self.client.config.application_id,
            source_component=getattr(record, 'source_component', self.source_component),
            timestamp=datetime.fromtimestamp(record.created, tz=timezone.utc),
            status=EventStatus.SUCCESS,
            metadata={}
        )

        # Extract standard fields from record
        if hasattr(record, 'input_text'):
            event.input_text = record.input_text
        if hasattr(record, 'output_text'):
            event.output_text = record.output_text
        if hasattr(record, 'token_count'):
            event.token_count = record.token_count
        if hasattr(record, 'latency_ms'):
            event.latency_ms = record.latency_ms
        if hasattr(record, 'cost'):
            event.cost = record.cost

        # Extract metadata and details from extra fields
        metadata = {}
        details = {}
        
        # Get all extra attributes from the record
        standard_fields = {
            'name', 'msg', 'args', 'levelname', 'levelno', 
            'pathname', 'filename', 'module', 'lineno', 'funcName',
            'created', 'msecs', 'relativeCreated', 'thread', 
            'threadName', 'processName', 'process', 'getMessage',
            'exc_info', 'exc_text', 'stack_info', 'event_type',
            'source_component', 'input_text', 'output_text', 
            'token_count', 'latency_ms', 'cost'
        }
        
        extra_fields = {k: v for k, v in record.__dict__.items() 
                       if k not in standard_fields}

        # Categorize extra fields
        for key, value in extra_fields.items():
            if key.startswith('meta_'):
                metadata[key[5:]] = value
            elif key.startswith('detail_'):
                details[key[7:]] = value
            elif key in ['provider', 'model_name', 'model', 'tool_name', 'action_type', 
                        'agent_name', 'endpoint', 'http_method', 'http_status_code',
                        'temperature', 'finish_reason', 'action', 'mcp_version',
                        'request_type', 'roundtrip_latency_ms', 'thought_process',
                        'selected_tool', 'target_model', 'policy']:
                details[key] = value
            else:
                metadata[key] = value

        # Set error status if this is an error log
        if record.levelno >= logging.ERROR:
            event.status = EventStatus.ERROR
            metadata['log_level'] = record.levelname
            metadata['error_message'] = record.getMessage()
            if record.exc_info:
                metadata['exception_type'] = record.exc_info[0].__name__
                metadata['exception_message'] = str(record.exc_info[1])

        # Set metadata if any
        if metadata:
            event.metadata = metadata
        
        return event, details

    def close(self):
        """Close handler and cleanup"""
        self._shutdown = True
        if self._processor_thread and self._processor_thread.is_alive():
            self._processor_thread.join(timeout=1.0)
        super().close()

    def __del__(self):
        """Cleanup on deletion"""
        if not self._shutdown:
            self.close()