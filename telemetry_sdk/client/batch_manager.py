"""
Fixed AutoBatchManager using lock-free approach with minor correctness updates.
"""

import asyncio
import threading
import time
from typing import List, Dict, Any, Optional, TYPE_CHECKING
from dataclasses import dataclass

from .models import (
    TelemetryEvent,
    EventIngestionRequest,
    BatchEventIngestionRequest,
    APIResponse,
)
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

    def __init__(self, client: "TelemetryClient"):
        self.client = client
        self._events: List[EventIngestionRequest] = []
        self._start_time: Optional[float] = None
        self._max_batch_size = client.config.batch_size
        self._max_payload_size = client.config.max_payload_size

    def add_event(self, event: TelemetryEvent, details: Optional[Dict[str, Any]] = None) -> "BatchManager":
        """Add an event to the batch."""
        if self.is_full():
            raise BatchError(f"Batch is full (max size: {self._max_batch_size})")

        request = EventIngestionRequest(event=event, details=details or {})

        # Enforce payload size limit
        if self._max_payload_size:
            request_size = len(str(request.to_dict()))
            if self.get_total_size() + request_size > self._max_payload_size:
                raise BatchError(
                    f"Adding event would exceed maximum payload size "
                    f"({self.get_total_size() + request_size} > {self._max_payload_size} bytes)"
                )

        self._events.append(request)
        return self

    def add_events(self, events: List[TelemetryEvent], details_list: Optional[List[Dict[str, Any]]] = None) -> "BatchManager":
        """Add multiple events to the batch."""
        if details_list and len(events) != len(details_list):
            raise ValidationError("Number of events and details must match")

        for i, event in enumerate(events):
            details = details_list[i] if details_list else None
            self.add_event(event, details)

        return self

    def clear(self) -> "BatchManager":
        """Clear all events."""
        self._events.clear()
        self._start_time = None
        return self

    def is_empty(self) -> bool:
        return len(self._events) == 0

    def is_full(self) -> bool:
        return len(self._events) >= self._max_batch_size

    def size(self) -> int:
        return len(self._events)

    def get_total_size(self) -> int:
        """Estimate total payload size (bytes)."""
        return sum(len(str(req.to_dict())) for req in self._events)

    def get_stats(self) -> BatchStats:
        """Return current batch statistics."""
        processing_time = 0
        if self._start_time:
            processing_time = int((time.time() - self._start_time) * 1000)
        return BatchStats(
            total_events=len(self._events),
            total_size_bytes=self.get_total_size(),
            processing_time_ms=processing_time,
        )

    async def send(self) -> APIResponse:
        """Send the batch."""
        if self.is_empty():
            return APIResponse.success_response({"message": "No events to send", "total_events": 0})

        self._start_time = time.time()

        try:
            batch_request = BatchEventIngestionRequest(events=self._events)
            # TODO: refactor to use public client.send_batch(batch_request)
            response = await self.client._send_batch_request(batch_request)

            stats = self.get_stats()
            stats.successful_events = len(self._events)

            response.data = response.data or {}
            response.data.update({"batch_stats": stats.__dict__})

            self.clear()
            return response

        except Exception as e:
            stats = self.get_stats()
            stats.failed_events = len(self._events)
            raise BatchError(f"Failed to send batch: {e}") from e

    async def send_and_clear(self) -> APIResponse:
        """Send and always clear."""
        try:
            return await self.send()
        finally:
            self.clear()

    def split_batch(self, max_size: Optional[int] = None) -> List["BatchManager"]:
        """Split large batch into smaller batches."""
        max_size = max_size or max(1, self._max_batch_size // 2)
        batches = []
        current = self._events.copy()

        while current:
            new_batch = BatchManager(self.client)
            new_batch._events = current[:max_size]
            batches.append(new_batch)
            current = current[max_size:]

        return batches

    async def send_in_chunks(self, chunk_size: Optional[int] = None) -> List[APIResponse]:
        """Send in smaller chunks if needed."""
        chunk_size = chunk_size or max(1, self._max_batch_size // 2)
        responses: List[APIResponse] = []

        for batch in self.split_batch(chunk_size):
            try:
                resp = await batch.send()
                responses.append(resp)
            except Exception as e:
                responses.append(APIResponse.error_response(str(e)))

        self.clear()
        return responses

    def validate_batch(self) -> List[str]:
        """Validate all events."""
        errors = []
        for i, request in enumerate(self._events):
            try:
                event = request.event
                if not event.event_id:
                    errors.append(f"Event {i}: Missing event_id")
                if not event.source_component:
                    errors.append(f"Event {i}: Missing source_component")
                if not event.application_id:
                    errors.append(f"Event {i}: Missing application_id")

                event_size = len(str(request.to_dict()))
                if self._max_payload_size and event_size > self._max_payload_size:
                    errors.append(
                        f"Event {i}: Size {event_size} > max {self._max_payload_size}"
                    )

            except Exception as e:
                errors.append(f"Event {i}: Validation error - {e}")
        return errors

    def __len__(self):
        return len(self._events)

    def __iter__(self):
        return iter(self._events)

    def __getitem__(self, idx: int) -> EventIngestionRequest:
        return self._events[idx]

class AutoBatchManager:
    """
    Dual-mode AutoBatchManager:
    - Async-safe for use inside event loops
    - Thread-safe background processing for sync spans
    """

    def __init__(self, client: "TelemetryClient", auto_send_timeout: Optional[float] = None):
        self.client = client
        self._auto_send_timeout = auto_send_timeout or client.config.batch_timeout
        self._last_send_time = time.time()

        # Internal queues
        self._event_queue = asyncio.Queue(maxsize=1000)
        self._batch = BatchManager(client)
        self._processing = False
        self._shutdown = False
        self._last_flush_check = time.time()
        self._flush_check_interval = 0.1

        # Background event loop (for threaded usage)
        self._background_loop: Optional[asyncio.AbstractEventLoop] = None
        self._background_thread: Optional[threading.Thread] = None
        self._start_background_loop()

    # --- Background loop setup -------------------------------------------------
    def _start_background_loop(self):
        """Starts a persistent background asyncio loop for safe cross-thread calls."""


        def run_loop():
            self._background_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._background_loop)
            self._background_loop.run_until_complete(self._background_task())

        self._background_thread = threading.Thread(target=run_loop, daemon=True)
        self._background_thread.start()

    async def _background_task(self):
        """Continuously process and flush queued events."""
        while not self._shutdown:
            try:
                await self._process_events()
                await asyncio.sleep(self._flush_check_interval)
            except Exception as e:
                if hasattr(self.client, "_logger"):
                    self.client._logger.warning(f"[AutoBatchManager] Background loop error: {e}")
                await asyncio.sleep(0.2)

    # --- Event handling --------------------------------------------------------
    async def add_event(self, event: TelemetryEvent, details: Optional[Dict[str, Any]] = None) -> Optional[APIResponse]:
        """Add event asynchronously without blocking."""
        try:
            #print(f"[AutoBatch] Adding event → {event.event_type} / {event.event_id}")
            self._event_queue.put_nowait((event, details or {}))
            #print(f"[AutoBatch] Queue size after add: {self._event_queue.qsize()}")
            
            if not self._processing:
                return await self._process_events()
            return None
        except asyncio.QueueFull:
            #print("[AutoBatch] Queue full; dropping event")
            self.client._logger.warning("AutoBatch queue full; dropping event.")
            return None
        except Exception as e:
            #print(f"[AutoBatch] Error adding event: {e}")
            self.client._logger.error(f"AutoBatch add_event error: {e}")
            return None


    async def _process_events(self) -> Optional[APIResponse]:
        """Drain events from queue and send in batches."""
        if self._processing:
            return None

        self._processing = True
        response = None

        try:
            processed = 0
            now = time.time()
            #print(f"[AutoBatch] Processing queue (initial size: {self._event_queue.qsize()})")

            while processed < self.client.config.batch_size and not self._event_queue.empty():
                try:
                    event, details = self._event_queue.get_nowait()
                    #print(f"[AutoBatch] → Dequeued event {event.event_id}")
                    self._batch.add_event(event, details)
                    processed += 1
                except asyncio.QueueEmpty:
                    #print("[AutoBatch] Queue empty mid-process")
                    break
                except Exception as e:
                    #print(f"[AutoBatch] Error processing event: {e}")
                    self.client._logger.warning(f"Queue event error: {e}")

            #print(f"[AutoBatch] Batch size after processing: {self._batch.size()}")

            should_send = (
                self._batch.is_full()
                or (now - self._last_send_time) >= self._auto_send_timeout
                or (processed > 0 and self._batch.size() >= self.client.config.batch_size // 2)
            )

            if should_send and not self._batch.is_empty():
                try:
                    #print(f"[AutoBatch] Sending batch of {self._batch.size()} events...")
                    response = await self._batch.send()
                    #print(f"[AutoBatch] ✅ Batch send complete.")
                    self._last_send_time = now
                except Exception as e:
                    #print(f"[AutoBatch] ❌ Batch send failed: {e}")
                    self.client._logger.warning(f"Batch send failed: {e}")
                    self._batch.clear()
                    self._last_send_time = now

        finally:
            await asyncio.sleep(0.01)
            self._processing = False
            #print(f"[AutoBatch] Processing finished. Queue size: {self._event_queue.qsize()} | Batch size: {self._batch.size()}")

        return response

    async def flush(self) -> Optional[APIResponse]:
        """Force flush all remaining events and batches."""
        #print(f"[AutoBatch] 🚀 Full flush start. Queue={self._event_queue.qsize()} Batch={self._batch.size()}")
        responses = []

        # Drain the queue completely
        while not self._event_queue.empty() or not self._batch.is_empty():
            await self._process_events()

            # If batch has events after draining — send it
            if not self._batch.is_empty():
                try:
                    #print(f"[AutoBatch] Flushing batch ({self._batch.size()} events)...")
                    resp = await self._batch.send()
                    self._last_send_time = time.time()
                    responses.append(resp)
                    #print("[AutoBatch] ✅ Batch flushed.")
                except Exception as e:
                    #print(f"[AutoBatch] ❌ Flush failed: {e}")
                    self.client._logger.warning(f"Flush failed: {e}")
                    self._batch.clear()

            # Give control back to event loop briefly
            await asyncio.sleep(0.05)

            # Stop if queue empty and batch empty
            if self._event_queue.empty() and self._batch.is_empty():
                break

        #print(f"[AutoBatch] 🧹 Flush complete. Remaining Queue={self._event_queue.qsize()} Batch={self._batch.size()}")
        return responses[-1] if responses else None



    async def _flush_internal(self) -> Optional[APIResponse]:
        """Internal flush implementation that always runs inside an event loop."""
        await self._process_events()
        if not self._batch.is_empty():
            try:
                resp = await self._batch.send()
                self._last_send_time = time.time()
                return resp
            except Exception as e:
                self.client._logger.warning(f"Flush failed: {e}")
                self._batch.clear()
        return None

    # --- Lifecycle -------------------------------------------------------------
    def stop(self):
        """Signal the background thread to stop."""
        self._shutdown = True
        if self._background_loop:
            try:
                self._background_loop.call_soon_threadsafe(self._background_loop.stop)
            except Exception:
                pass
        if self._background_thread and self._background_thread.is_alive():
            self._background_thread.join(timeout=2.0)

    def get_pending_count(self) -> int:
        return self._event_queue.qsize() + self._batch.size()

    def get_stats(self) -> BatchStats:
        return self._batch.get_stats()
