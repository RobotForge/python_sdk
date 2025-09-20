"""
Logging integration for telemetry SDK
Allows users to send telemetry events through standard Python logging
"""

import logging
import asyncio
import threading
from typing import Dict, Optional
from queue import Queue
from datetime import datetime, timezone

from client.models import EventStatus, EventType, TelemetryEvent
from client.telemetry_client import TelemetryClient



class TelemetryHandler(logging.Handler):
    """Logging handler that converts log records to telemetry events"""
    
    def __init__(
        self,
        client: TelemetryClient,
        level: int = logging.INFO,
        event_type_mapping: Optional[Dict[str, EventType]] = None,
        source_component: str = "logging"
    ):
        super().__init__(level)
        self.client = client
        self.source_component = source_component
        
        # Default mapping of log record names to event types
        self.event_type_mapping = event_type_mapping or {
            'model_call': EventType.MODEL_CALL,
            'tool_execution': EventType.TOOL_EXECUTION,
            'mcp_event': EventType.MCP_EVENT,
            'agent_action': EventType.AGENT_ACTION,
        }
        
        # Queue for async processing
        self._event_queue = Queue()
        self._processor_thread = None
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
                    event = TelemetryEvent(**event_data['event'])
                    await self.client._send_event(event, event_data['details'])
                else:
                    await asyncio.sleep(0.1)
            except Exception as e:
                # Don't let telemetry errors break the application
                pass

    def emit(self, record: logging.LogRecord):
        """Convert log record to telemetry event"""
        try:
            # Determine event type
            event_type = self._determine_event_type(record)
            if not event_type:
                return  # Skip if can't determine event type

            # Extract telemetry data from log record
            event_data, details = self._extract_telemetry_data(record, event_type)
            
            # Queue for async processing
            self._event_queue.put({
                'event': event_data,
                'details': details
            })
            
        except Exception:
            # Never let telemetry break the application
            self.handleError(record)

    def _determine_event_type(self, record: logging.LogRecord) -> Optional[EventType]:
        """Determine event type from log record"""
        # Check if event type is explicitly specified in the log message
        if hasattr(record, 'event_type'):
            return getattr(EventType, record.event_type.upper(), None)
        
        # Check message content
        message = record.getMessage().lower()
        for key, event_type in self.event_type_mapping.items():
            if key in message:
                return event_type
        
        # Check logger name
        logger_name = record.name.lower()
        for key, event_type in self.event_type_mapping.items():
            if key in logger_name:
                return event_type
        
        # Default fallback
        return EventType.AGENT_ACTION

    def _extract_telemetry_data(self, record: logging.LogRecord, event_type: EventType) -> tuple:
        """Extract telemetry data from log record"""
        
        # Base event data
        event_data = {
            'event_id': f"evt_{record.created}_{threading.get_ident()}",
            'event_type': event_type,
            'session_id': self.client.session_id,
            'tenant_id': self.client.tenant_id,
            'project_id': self.client.project_id,
            'user_id': self.client.user_id,
            'application_id': self.client.application_id,
            'source_component': getattr(record, 'source_component', self.source_component),
            'timestamp': datetime.fromtimestamp(record.created, tz=timezone.utc),
            'status': EventStatus.SUCCESS,
            'metadata': {}
        }

        # Extract standard fields from record
        if hasattr(record, 'input_text'):
            event_data['input_text'] = record.input_text
        if hasattr(record, 'output_text'):
            event_data['output_text'] = record.output_text
        if hasattr(record, 'token_count'):
            event_data['token_count'] = record.token_count
        if hasattr(record, 'latency_ms'):
            event_data['latency_ms'] = record.latency_ms
        if hasattr(record, 'cost'):
            event_data['cost'] = record.cost

        # Extract metadata from extra fields
        metadata = {}
        details = {}
        
        # Get all extra attributes from the record
        extra_fields = {k: v for k, v in record.__dict__.items() 
                       if k not in ['name', 'msg', 'args', 'levelname', 'levelno', 
                                  'pathname', 'filename', 'module', 'lineno', 'funcName',
                                  'created', 'msecs', 'relativeCreated', 'thread', 
                                  'threadName', 'processName', 'process', 'getMessage',
                                  'exc_info', 'exc_text', 'stack_info']}

        # Categorize extra fields
        for key, value in extra_fields.items():
            if key.startswith('meta_'):
                metadata[key[5:]] = value
            elif key.startswith('detail_'):
                details[key[7:]] = value
            elif key in ['provider', 'model_name', 'model', 'tool_name', 'action_type', 
                        'agent_name', 'endpoint', 'http_method', 'http_status_code']:
                details[key] = value
            else:
                metadata[key] = value

        # Set error status if this is an error log
        if record.levelno >= logging.ERROR:
            event_data['status'] = EventStatus.ERROR
            metadata['error_message'] = record.getMessage()
            if record.exc_info:
                metadata['exception'] = str(record.exc_info[1])

        event_data['metadata'] = metadata if metadata else None
        
        return event_data, details

    def close(self):
        """Close handler and cleanup"""
        self._shutdown = True
        if self._processor_thread and self._processor_thread.is_alive():
            self._processor_thread.join(timeout=1.0)
        super().close()

