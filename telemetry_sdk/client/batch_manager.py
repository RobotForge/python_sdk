"""
Fixed AutoBatchManager using lock-free approach
File: telemetry_sdk/client/batch_manager.py

Only this file needs to be updated - keeps the same API but removes asyncio.Lock
"""

import asyncio
import time
from typing import List, Dict, Any, Optional, TYPE_CHECKING
from dataclasses import dataclass

from .models import TelemetryEvent, EventIngestionRequest, BatchEventIngestionRequest, APIResponse
from ..utils.exceptions import BatchError, ValidationError

if TYPE_CHECKING:
    from .telemetry_client import TelemetryClient


@dataclass
class BatchStats:
    """Statistics for batch operations"""
    total_events: int = 0
    successful_events: int = 0
    failed_events: int = 0
    total_size_bytes: int = 0
    processing_time_ms: int = 0


class BatchManager:
    """Manages batching and bulk sending of telemetry events"""
    
    def __init__(self, client: 'TelemetryClient'):
        self.client = client
        self._events: List[EventIngestionRequest] = []
        self._start_time: Optional[float] = None
        self._max_batch_size = client.config.batch_size
        self._max_payload_size = client.config.max_payload_size

    def add_event(
        self, 
        event: TelemetryEvent, 
        details: Optional[Dict[str, Any]] = None
    ) -> 'BatchManager':
        """Add an event to the batch"""
        if self.is_full():
            raise BatchError(f"Batch is full (max size: {self._max_batch_size})")
        
        request = EventIngestionRequest(event=event, details=details or {})
        
        # Check payload size if configured
        if self._max_payload_size:
            request_size = len(str(request.to_dict()))
            current_size = self.get_total_size()
            
            if current_size + request_size > self._max_payload_size:
                raise BatchError(
                    f"Adding event would exceed maximum payload size "
                    f"({current_size + request_size} > {self._max_payload_size} bytes)"
                )
        
        self._events.append(request)
        return self

    def add_events(
        self, 
        events: List[TelemetryEvent], 
        details_list: Optional[List[Dict[str, Any]]] = None
    ) -> 'BatchManager':
        """Add multiple events to the batch"""
        if details_list and len(events) != len(details_list):
            raise ValidationError("Number of events and details must match")
        
        for i, event in enumerate(events):
            details = details_list[i] if details_list else None
            self.add_event(event, details)
        
        return self

    def remove_event(self, index: int) -> 'BatchManager':
        """Remove an event from the batch by index"""
        if 0 <= index < len(self._events):
            self._events.pop(index)
        else:
            raise IndexError(f"Invalid event index: {index}")
        return self

    def clear(self) -> 'BatchManager':
        """Clear all events from the batch"""
        self._events.clear()
        self._start_time = None
        return self

    def is_empty(self) -> bool:
        """Check if the batch is empty"""
        return len(self._events) == 0

    def is_full(self) -> bool:
        """Check if the batch is at maximum capacity"""
        return len(self._events) >= self._max_batch_size

    def size(self) -> int:
        """Get the number of events in the batch"""
        return len(self._events)

    def get_total_size(self) -> int:
        """Get the total size in bytes of all events in the batch"""
        total_size = 0
        for request in self._events:
            total_size += len(str(request.to_dict()))
        return total_size

    def get_events(self) -> List[EventIngestionRequest]:
        """Get a copy of all events in the batch"""
        return self._events.copy()

    def get_stats(self) -> BatchStats:
        """Get statistics about the current batch"""
        processing_time = 0
        if self._start_time:
            processing_time = int((time.time() - self._start_time) * 1000)
        
        return BatchStats(
            total_events=len(self._events),
            total_size_bytes=self.get_total_size(),
            processing_time_ms=processing_time
        )

    async def send(self) -> APIResponse:
        """Send all events in the batch"""
        if self.is_empty():
            return APIResponse.success_response(
                {"message": "No events to send", "total_events": 0}
            )
        
        self._start_time = time.time()
        
        try:
            # Create batch request
            batch_request = BatchEventIngestionRequest(events=self._events)
            
            # Send via client
            response = await self.client._send_batch_request(batch_request)
            
            # Update stats
            stats = self.get_stats()
            stats.successful_events = len(self._events)
            
            # Clear the batch after successful send
            self.clear()
            
            return response
            
        except Exception as e:
            # Update stats with failure
            stats = self.get_stats()
            stats.failed_events = len(self._events)
            
            raise BatchError(f"Failed to send batch: {str(e)}") from e

    async def send_and_clear(self) -> APIResponse:
        """Send all events and clear the batch regardless of success/failure"""
        try:
            return await self.send()
        finally:
            self.clear()

    def split_batch(self, max_size: Optional[int] = None) -> List['BatchManager']:
        """Split the current batch into smaller batches"""
        if max_size is None:
            max_size = self._max_batch_size // 2
        
        if max_size <= 0:
            raise ValidationError("Max size must be greater than 0")
        
        batches = []
        current_events = self._events.copy()
        
        while current_events:
            # Create new batch manager
            new_batch = BatchManager(self.client)
            
            # Add events up to max_size
            events_to_add = current_events[:max_size]
            for request in events_to_add:
                new_batch._events.append(request)
            
            batches.append(new_batch)
            current_events = current_events[max_size:]
        
        return batches

    async def send_in_chunks(self, chunk_size: Optional[int] = None) -> List[APIResponse]:
        """Send the batch in smaller chunks"""
        if chunk_size is None:
            chunk_size = self._max_batch_size // 2
        
        batches = self.split_batch(chunk_size)
        responses = []
        
        for batch in batches:
            try:
                response = await batch.send()
                responses.append(response)
            except Exception as e:
                # Continue with other batches even if one fails
                error_response = APIResponse.error_response(str(e))
                responses.append(error_response)
        
        # Clear the original batch
        self.clear()
        
        return responses

    def validate_batch(self) -> List[str]:
        """Validate all events in the batch and return list of validation errors"""
        errors = []
        
        for i, request in enumerate(self._events):
            try:
                # Basic validation
                event = request.event
                if not event.event_id:
                    errors.append(f"Event {i}: Missing event_id")
                if not event.source_component:
                    errors.append(f"Event {i}: Missing source_component")
                if not event.project_id:
                    errors.append(f"Event {i}: Missing project_id")
                
                # Size validation
                event_size = len(str(request.to_dict()))
                if self._max_payload_size and event_size > self._max_payload_size:
                    errors.append(
                        f"Event {i}: Size ({event_size} bytes) exceeds maximum "
                        f"({self._max_payload_size} bytes)"
                    )
                    
            except Exception as e:
                errors.append(f"Event {i}: Validation error - {str(e)}")
        
        return errors

    def __len__(self) -> int:
        """Return the number of events in the batch"""
        return len(self._events)

    def __bool__(self) -> bool:
        """Return True if the batch has events"""
        return not self.is_empty()

    def __iter__(self):
        """Iterate over events in the batch"""
        return iter(self._events)

    def __getitem__(self, index: int) -> EventIngestionRequest:
        """Get an event by index"""
        return self._events[index]


class AutoBatchManager:
    """
    Lock-free automatic batch manager using asyncio.Queue instead of locks
    Keeps the same API but removes all asyncio.Lock usage
    """
    
    def __init__(
        self, 
        client: 'TelemetryClient',
        auto_send_timeout: Optional[float] = None
    ):
        self.client = client
        self._auto_send_timeout = auto_send_timeout or client.config.batch_timeout
        self._last_send_time = time.time()
        
        # Use asyncio.Queue instead of locks - this is the key change!
        self._event_queue = asyncio.Queue(maxsize=1000)  # Bounded queue
        self._batch = BatchManager(client)
        self._processing = False  # Simple flag to prevent concurrent processing
        
        # Track timing for auto-flush
        self._last_flush_check = time.time()
        self._flush_check_interval = 0.1  # Check for flush every 100ms

    async def add_event(
        self, 
        event: TelemetryEvent, 
        details: Optional[Dict[str, Any]] = None
    ) -> Optional[APIResponse]:
        """Add event using queue instead of locks"""
        try:
            # Add to queue without blocking
            event_data = (event, details or {})
            self._event_queue.put_nowait(event_data)
            
            # Process events if not already processing
            if not self._processing:
                return await self._process_events()
            
            return None
            
        except asyncio.QueueFull:
            # If queue is full, drop the event rather than block
            self.client._logger.warning("AutoBatch queue full, dropping event to prevent blocking")
            return None
        except Exception as e:
            self.client._logger.error(f"AutoBatchManager add_event error: {e}")
            return None

    async def _process_events(self) -> Optional[APIResponse]:
        """Process events from queue without locks"""
        if self._processing:
            return None
            
        self._processing = True
        response = None
        
        try:
            # Process events from queue in batches
            events_processed = 0
            current_time = time.time()
            
            # Process up to batch_size events or until queue is empty
            while events_processed < self.client.config.batch_size and not self._event_queue.empty():
                try:
                    # Get event without blocking
                    event, details = self._event_queue.get_nowait()
                    self._batch.add_event(event, details)
                    events_processed += 1
                except asyncio.QueueEmpty:
                    break
                except Exception as e:
                    self.client._logger.warning(f"Error processing queued event: {e}")
                    continue
            
            # Check if we should send batch
            should_send = (
                self._batch.is_full() or 
                (current_time - self._last_send_time) >= self._auto_send_timeout or
                (events_processed > 0 and self._batch.size() >= self.client.config.batch_size // 2)
            )
            
            if should_send and not self._batch.is_empty():
                try:
                    response = await self._batch.send()
                    self._last_send_time = current_time
                except Exception as e:
                    # Don't let batch errors break the application
                    self.client._logger.warning(f"Batch send failed: {e}")
                    self._batch.clear()
                    self._last_send_time = current_time
                    
        finally:
            self._processing = False
            
        return response

    async def flush(self) -> Optional[APIResponse]:
        """Flush remaining events without locks"""
        if self._processing:
            # If already processing, just wait a bit and return
            await asyncio.sleep(0.01)
            return None
        
        # Process any remaining queued events first
        await self._process_events()
        
        # Then flush the batch
        if not self._batch.is_empty():
            try:
                response = await self._batch.send()
                self._last_send_time = time.time()
                return response
            except Exception as e:
                self.client._logger.warning(f"Flush failed: {e}")
                self._batch.clear()
                return None
        
        return None

    def get_pending_count(self) -> int:
        """Get the number of pending events"""
        return self._event_queue.qsize() + self._batch.size()

    def get_stats(self) -> BatchStats:
        """Get current batch statistics"""
        return self._batch.get_stats()