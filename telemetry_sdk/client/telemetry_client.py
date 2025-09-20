"""
Telemetry SDK for capturing and sending telemetry events
Multi-layered approach supporting decorators, context managers, auto-instrumentation, and manual control
"""

import asyncio
import uuid
import time
import logging
from typing import Dict, Any, Optional, List, AsyncGenerator
from contextlib import asynccontextmanager
from functools import wraps
import aiohttp
import threading
from queue import Queue

from client.batch_manager import BatchManager
from client.event_builder import EventBuilder, EventStatus, EventType, TelemetryEvent
from client.trace_context import TraceContext





class TelemetryClient:
    """Main telemetry client supporting multiple integration patterns"""
    
    def __init__(
        self,
        api_key: str,
        endpoint: str,
        project_id: str,
        tenant_id: str = "default",
        user_id: str = "default",
        application_id: str = "default",
        session_id: Optional[str] = None,
        auto_send: bool = True,
        batch_size: int = 50,
        batch_timeout: float = 5.0,
        pii_scrubbing: bool = False,
        max_payload_size: int = 100_000,
        retry_attempts: int = 3
    ):
        self.api_key = api_key
        self.endpoint = endpoint.rstrip('/')
        self.project_id = project_id
        self.tenant_id = tenant_id
        self.user_id = user_id
        self.application_id = application_id
        self.session_id = session_id or f"sess_{uuid.uuid4().hex[:12]}"
        
        # Configuration
        self.auto_send = auto_send
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.pii_scrubbing = pii_scrubbing
        self.max_payload_size = max_payload_size
        self.retry_attempts = retry_attempts
        
        # Internal state
        self._session = None
        self._batch_queue = Queue()
        self._background_task = None
        self._shutdown = False
        
        # Headers for API requests
        self._headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        
        if auto_send:
            self._start_background_sender()

    def _start_background_sender(self):
        """Start background thread for batch sending"""
        def background_sender():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._batch_sender_loop())
            loop.close()
        
        self._background_task = threading.Thread(target=background_sender, daemon=True)
        self._background_task.start()

    async def _batch_sender_loop(self):
        """Background loop for sending batched events"""
        batch = []
        last_send = time.time()
        
        while not self._shutdown:
            try:
                # Collect events for batching
                while len(batch) < self.batch_size and (time.time() - last_send) < self.batch_timeout:
                    try:
                        event_data = self._batch_queue.get(timeout=0.1)
                        batch.append(event_data)
                    except:
                        continue
                
                # Send batch if we have events
                if batch:
                    await self._send_batch_internal(batch)
                    batch.clear()
                    last_send = time.time()
                    
            except Exception as e:
                logging.error(f"Telemetry batch sender error: {e}")
                await asyncio.sleep(1)

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session"""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            self._session = aiohttp.ClientSession(
                headers=self._headers,
                timeout=timeout
            )
        return self._session

    async def _send_event(self, event: TelemetryEvent, details: Optional[Dict[str, Any]] = None):
        """Send a single event"""
        if self.auto_send:
            # Add to batch queue
            self._batch_queue.put({
                "event": event.to_dict(),
                "details": details or {}
            })
        else:
            # Send immediately
            payload = {
                "event": event.to_dict(),
                "details": details or {}
            }
            await self._send_single_event(payload)

    async def _send_single_event(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Send single event via API"""
        session = await self._get_session()
        url = f"{self.endpoint}/api/v1/events"
        
        for attempt in range(self.retry_attempts):
            try:
                async with session.post(url, json=payload) as response:
                    if response.status == 201:
                        return await response.json()
                    else:
                        error_text = await response.text()
                        raise Exception(f"API error {response.status}: {error_text}")
            except Exception as e:
                if attempt == self.retry_attempts - 1:
                    logging.error(f"Failed to send telemetry event after {self.retry_attempts} attempts: {e}")
                    raise
                await asyncio.sleep(2 ** attempt)

    async def _send_batch(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Send batch of events via API"""
        session = await self._get_session()
        url = f"{self.endpoint}/api/v1/events/batch"
        
        for attempt in range(self.retry_attempts):
            try:
                async with session.post(url, json=payload) as response:
                    if response.status == 201:
                        return await response.json()
                    else:
                        error_text = await response.text()
                        raise Exception(f"API error {response.status}: {error_text}")
            except Exception as e:
                if attempt == self.retry_attempts - 1:
                    logging.error(f"Failed to send telemetry batch after {self.retry_attempts} attempts: {e}")
                    raise
                await asyncio.sleep(2 ** attempt)

    async def _send_batch_internal(self, events: List[Dict[str, Any]]):
        """Internal method to send batch"""
        if not events:
            return
        
        payload = {"events": events}
        try:
            await self._send_batch(payload)
        except Exception as e:
            logging.error(f"Failed to send telemetry batch: {e}")

    # Context Manager Methods
    @asynccontextmanager
    async def trace_model_call(self, **kwargs) -> AsyncGenerator[EventBuilder, None]:
        """Context manager for tracing model calls"""
        async with TraceContext(self, EventType.MODEL_CALL, kwargs.get('source_component', 'model_call'), **kwargs) as builder:
            yield builder

    @asynccontextmanager
    async def trace_tool_execution(self, tool_name: str, **kwargs) -> AsyncGenerator[EventBuilder, None]:
        """Context manager for tracing tool executions"""
        kwargs['tool_name'] = tool_name
        async with TraceContext(self, EventType.TOOL_EXECUTION, kwargs.get('source_component', tool_name), **kwargs) as builder:
            yield builder

    @asynccontextmanager
    async def trace_agent_action(self, action_type: str, **kwargs) -> AsyncGenerator[EventBuilder, None]:
        """Context manager for tracing agent actions"""
        kwargs['action_type'] = action_type
        async with TraceContext(self, EventType.AGENT_ACTION, kwargs.get('source_component', 'agent'), **kwargs) as builder:
            yield builder

    @asynccontextmanager
    async def trace_mcp_event(self, **kwargs) -> AsyncGenerator[EventBuilder, None]:
        """Context manager for tracing MCP events"""
        async with TraceContext(self, EventType.MCP_EVENT, kwargs.get('source_component', 'mcp'), **kwargs) as builder:
            yield builder

    # Builder Methods
    def create_event(self, event_type: EventType, source_component: str) -> EventBuilder:
        """Create an event builder"""
        return EventBuilder(self, event_type, source_component)

    def create_batch(self) -> BatchManager:
        """Create a batch manager"""
        return BatchManager(self)

    # Decorator Methods
    def trace_model_call_decorator(self, provider: str = None, model: str = None, **kwargs):
        """Decorator for tracing model calls"""
        def decorator(func):
            if asyncio.iscoroutinefunction(func):
                @wraps(func)
                async def async_wrapper(*args, **func_kwargs):
                    source_component = kwargs.get('source_component', func.__name__)
                    async with self.trace_model_call(
                        provider=provider, 
                        model=model, 
                        source_component=source_component,
                        **kwargs
                    ) as span:
                        try:
                            result = await func(*args, **func_kwargs)
                            # Try to extract token info from result if it's a common LLM response format
                            if hasattr(result, 'usage'):
                                span.set_tokens(getattr(result.usage, 'total_tokens', None))
                            return result
                        except Exception as e:
                            span.set_status(EventStatus.ERROR)
                            raise
                return async_wrapper
            else:
                @wraps(func)
                def sync_wrapper(*args, **func_kwargs):
                    # For sync functions, we need to handle this differently
                    # This is a simplified version - in practice you'd want to run the async context in a thread
                    result = func(*args, **func_kwargs)
                    # Create event synchronously (would need sync version of client)
                    return result
                return sync_wrapper
        return decorator

    def trace_tool_execution_decorator(self, tool_name: str, **kwargs):
        """Decorator for tracing tool executions"""
        def decorator(func):
            if asyncio.iscoroutinefunction(func):
                @wraps(func)
                async def async_wrapper(*args, **func_kwargs):
                    source_component = kwargs.get('source_component', tool_name)
                    async with self.trace_tool_execution(
                        tool_name=tool_name,
                        source_component=source_component,
                        **kwargs
                    ) as span:
                        try:
                            result = await func(*args, **func_kwargs)
                            return result
                        except Exception as e:
                            span.set_status(EventStatus.ERROR)
                            raise
                return async_wrapper
            else:
                @wraps(func)
                def sync_wrapper(*args, **func_kwargs):
                    return func(*args, **func_kwargs)
                return sync_wrapper
        return decorator

    def trace_agent_action_decorator(self, action_type: str, **kwargs):
        """Decorator for tracing agent actions"""
        def decorator(func):
            if asyncio.iscoroutinefunction(func):
                @wraps(func)
                async def async_wrapper(*args, **func_kwargs):
                    source_component = kwargs.get('source_component', 'agent')
                    async with self.trace_agent_action(
                        action_type=action_type,
                        source_component=source_component,
                        **kwargs
                    ) as span:
                        try:
                            result = await func(*args, **func_kwargs)
                            return result
                        except Exception as e:
                            span.set_status(EventStatus.ERROR)
                            raise
                return async_wrapper
            else:
                @wraps(func)
                def sync_wrapper(*args, **func_kwargs):
                    return func(*args, **func_kwargs)
                return sync_wrapper
        return decorator

    async def flush(self):
        """Flush any pending events"""
        if self.auto_send:
            # Wait for background queue to empty
            while not self._batch_queue.empty():
                await asyncio.sleep(0.1)
        
    async def close(self):
        """Close the client and cleanup resources"""
        self._shutdown = True
        await self.flush()
        
        if self._session and not self._session.closed:
            await self._session.close()
        
        if self._background_task and self._background_task.is_alive():
            self._background_task.join(timeout=1.0)

    def __del__(self):
        """Cleanup on deletion"""
        if self._session and not self._session.closed:
            # Can't await in __del__, so we schedule it
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self._session.close())
            except:
                pass