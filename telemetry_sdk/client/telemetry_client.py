"""
Main Telemetry Client implementation with all integration patterns
"""

import asyncio
import json
import logging
import threading
import time
from contextlib import asynccontextmanager
from functools import wraps
from typing import Dict, Any, Optional, List, AsyncGenerator, Callable
from queue import Queue

import aiohttp

from .models import (
    TelemetryConfig, TelemetryEvent, EventType, EventStatus, 
    EventIngestionRequest, BatchEventIngestionRequest, APIResponse
)
from .event_builder import EventBuilder, ModelCallEventBuilder, ToolExecutionEventBuilder, AgentActionEventBuilder
from .trace_context import TraceContext, SyncTraceContext
from .batch_manager import BatchManager, AutoBatchManager
from ..utils.exceptions import (
    TelemetrySDKError, NetworkError, AuthenticationError, 
    TimeoutError, PayloadTooLargeError, RateLimitError
)


class TelemetryClient:
    """
    Main telemetry client supporting multiple integration patterns:
    - Context managers for explicit tracing
    - Decorators for automatic function tracing  
    - Manual event creation and batching
    - Auto-instrumentation hooks
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        project_id: Optional[str] = None,
        config: Optional[TelemetryConfig] = None,
        **kwargs
    ):
        # Initialize configuration
        if config:
            self.config = config
        else:
            config_params = {
                'api_key': api_key,
                'endpoint': endpoint, 
                'project_id': project_id,
                **kwargs
            }
            # Remove None values
            config_params = {k: v for k, v in config_params.items() if v is not None}
            self.config = TelemetryConfig(**config_params)
        
        # Initialize internal state
        self._session: Optional[aiohttp.ClientSession] = None
        self._logger = logging.getLogger(__name__)
        self._shutdown = False
        
        # Headers for API requests
        self._headers = {
            'Authorization': f'Bearer {self.config.api_key}',
            'Content-Type': 'application/json',
            'User-Agent': f'telemetry-sdk-python/1.0.0'
        }
        
        # Auto-batching setup
        self._auto_batch_manager: Optional[AutoBatchManager] = None
        self._sync_event_queue: Queue = Queue()
        self._background_task: Optional[threading.Thread] = None
        
        if self.config.auto_send:
            self._setup_auto_batching()

    def _setup_auto_batching(self):
        """Setup automatic batching and background sending"""
        self._auto_batch_manager = AutoBatchManager(self)
        
        # Start background thread for sync event processing
        def background_processor():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._process_background_events())
            loop.close()
        
        self._background_task = threading.Thread(target=background_processor, daemon=True)
        self._background_task.start()

    async def _process_background_events(self):
        """Background loop for processing sync events and auto-batching"""
        while not self._shutdown:
            try:
                # Process sync events from queue
                while not self._sync_event_queue.empty():
                    try:
                        event_data = self._sync_event_queue.get_nowait()
                        event = event_data['event']
                        details = event_data['details']
                        
                        if self._auto_batch_manager:
                            await self._auto_batch_manager.add_event(event, details)
                    except:
                        continue
                
                # Flush auto-batch if timeout reached
                if self._auto_batch_manager:
                    await self._auto_batch_manager.flush()
                
                await asyncio.sleep(0.1)
                
            except Exception as e:
                self._logger.error(f"Background processing error: {e}")
                await asyncio.sleep(1)

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session with proper configuration"""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.config.request_timeout)
            connector = aiohttp.TCPConnector(
                limit=100,  # Connection pool limit
                limit_per_host=30,
                ttl_dns_cache=300,
                use_dns_cache=True
            )
            
            self._session = aiohttp.ClientSession(
                headers=self._headers,
                timeout=timeout,
                connector=connector,
                json_serialize=json.dumps
            )
        
        return self._session

    async def _send_event_request(self, request: EventIngestionRequest) -> APIResponse:
        """Send a single event request"""
        if self.config.auto_send and self._auto_batch_manager:
            # Use auto-batching
            response = await self._auto_batch_manager.add_event(request.event, request.details)
            if response:
                return response
            else:
                # Event was batched, return success
                return APIResponse.success_response({
                    "event_id": request.event.event_id,
                    "status": "queued_for_batch"
                })
        else:
            # Send immediately
            return await self._send_single_event(request)

    async def _send_single_event(self, request: EventIngestionRequest) -> APIResponse:
        """Send a single event immediately"""
        session = await self._get_session()
        url = f"{self.config.endpoint}/api/v1/events"
        payload = request.to_dict()
        
        return await self._make_request('POST', url, payload)

    async def _send_batch_request(self, request: BatchEventIngestionRequest) -> APIResponse:
        """Send a batch of events"""
        session = await self._get_session()
        url = f"{self.config.endpoint}/api/v1/events/batch"
        payload = request.to_dict()
        
        return await self._make_request('POST', url, payload)

    async def _make_request(self, method: str, url: str, payload: Dict[str, Any]) -> APIResponse:
        """Make HTTP request with retries and error handling"""
        session = await self._get_session()
        
        for attempt in range(self.config.retry_attempts + 1):
            try:
                async with session.request(method, url, json=payload) as response:
                    response_data = await response.json()
                    
                    if response.status == 201 or response.status == 200:
                        return APIResponse.success_response(response_data, response.status)
                    elif response.status == 401:
                        raise AuthenticationError("Invalid API key or authentication failed")
                    elif response.status == 413:
                        raise PayloadTooLargeError("Request payload too large")
                    elif response.status == 429:
                        raise RateLimitError("Rate limit exceeded")
                    else:
                        error_msg = response_data.get('detail', f'HTTP {response.status}')
                        if attempt == self.config.retry_attempts:
                            return APIResponse.error_response(error_msg, response.status)
                        # Continue to retry
                        
            except asyncio.TimeoutError:
                if attempt == self.config.retry_attempts:
                    raise TimeoutError("Request timed out")
            except aiohttp.ClientError as e:
                if attempt == self.config.retry_attempts:
                    raise NetworkError(f"Network error: {str(e)}")
            except Exception as e:
                if attempt == self.config.retry_attempts:
                    raise TelemetrySDKError(f"Unexpected error: {str(e)}")
            
            # Exponential backoff for retries
            if attempt < self.config.retry_attempts:
                await asyncio.sleep(2 ** attempt)
        
        return APIResponse.error_response("Max retries exceeded")

    def _queue_sync_event(self, event: TelemetryEvent, details: Dict[str, Any]):
        """Queue event from sync context for async processing"""
        try:
            self._sync_event_queue.put({
                'event': event,
                'details': details
            }, block=False)
        except:
            # Queue is full, drop the event to avoid blocking
            self._logger.warning("Sync event queue full, dropping event")

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
        source_component = kwargs.get('source_component', tool_name)
        async with TraceContext(self, EventType.TOOL_EXECUTION, source_component, **kwargs) as builder:
            yield builder

    @asynccontextmanager
    async def trace_agent_action(self, action_type: str, **kwargs) -> AsyncGenerator[EventBuilder, None]:
        """Context manager for tracing agent actions"""
        kwargs['action_type'] = action_type
        source_component = kwargs.get('source_component', 'agent')
        async with TraceContext(self, EventType.AGENT_ACTION, source_component, **kwargs) as builder:
            yield builder

    @asynccontextmanager
    async def trace_mcp_event(self, **kwargs) -> AsyncGenerator[EventBuilder, None]:
        """Context manager for tracing MCP events"""
        source_component = kwargs.get('source_component', 'mcp')
        async with TraceContext(self, EventType.MCP_EVENT, source_component, **kwargs) as builder:
            yield builder

    # Sync Context Manager Methods
    def trace_model_call_sync(self, **kwargs):
        """Synchronous context manager for tracing model calls"""
        source_component = kwargs.get('source_component', 'model_call')
        return SyncTraceContext(self, EventType.MODEL_CALL, source_component, **kwargs)

    def trace_tool_execution_sync(self, tool_name: str, **kwargs):
        """Synchronous context manager for tracing tool executions"""
        kwargs['tool_name'] = tool_name
        source_component = kwargs.get('source_component', tool_name)
        return SyncTraceContext(self, EventType.TOOL_EXECUTION, source_component, **kwargs)

    def trace_agent_action_sync(self, action_type: str, **kwargs):
        """Synchronous context manager for tracing agent actions"""
        kwargs['action_type'] = action_type
        source_component = kwargs.get('source_component', 'agent')
        return SyncTraceContext(self, EventType.AGENT_ACTION, source_component, **kwargs)

    # Event Builder Factory Methods
    def create_model_call_event(self, source_component: str = "model_call") -> ModelCallEventBuilder:
        """Create a model call event builder"""
        return ModelCallEventBuilder(self, source_component)

    def create_tool_execution_event(self, tool_name: str, source_component: str = None) -> ToolExecutionEventBuilder:
        """Create a tool execution event builder"""
        return ToolExecutionEventBuilder(self, tool_name, source_component)

    def create_agent_action_event(self, action_type: str, source_component: str = "agent") -> AgentActionEventBuilder:
        """Create an agent action event builder"""
        return AgentActionEventBuilder(self, action_type, source_component)

    def create_event(self, event_type: EventType, source_component: str) -> EventBuilder:
        """Create a generic event builder"""
        return EventBuilder(self, event_type, source_component)

    # Batch Management Methods
    def create_batch(self) -> BatchManager:
        """Create a new batch manager"""
        return BatchManager(self)

    async def send_events(self, events: List[TelemetryEvent], details_list: Optional[List[Dict[str, Any]]] = None) -> APIResponse:
        """Send multiple events as a batch"""
        batch = self.create_batch()
        batch.add_events(events, details_list)
        return await batch.send()

    # Decorator Methods
    def trace_model_call_decorator(self, provider: str = None, model: str = None, **kwargs):
        """Decorator for automatically tracing model calls"""
        def decorator(func: Callable):
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
                            
                            # Try to extract common response patterns
                            if hasattr(result, 'usage'):
                                if hasattr(result.usage, 'total_tokens'):
                                    span.set_tokens(result.usage.total_tokens)
                            
                            if hasattr(result, 'choices') and result.choices:
                                if hasattr(result.choices[0], 'message'):
                                    span.set_output(str(result.choices[0].message.content)[:1000])
                                elif hasattr(result.choices[0], 'text'):
                                    span.set_output(str(result.choices[0].text)[:1000])
                            
                            return result
                            
                        except Exception as e:
                            span.set_status(EventStatus.ERROR)
                            span.set_error(e)
                            raise
                
                return async_wrapper
            else:
                @wraps(func)
                def sync_wrapper(*args, **func_kwargs):
                    source_component = kwargs.get('source_component', func.__name__)
                    
                    with self.trace_model_call_sync(
                        provider=provider,
                        model=model,
                        source_component=source_component,
                        **kwargs
                    ) as span:
                        try:
                            result = func(*args, **func_kwargs)
                            
                            # Try to extract common response patterns
                            if hasattr(result, 'usage'):
                                if hasattr(result.usage, 'total_tokens'):
                                    span.set_tokens(result.usage.total_tokens)
                            
                            return result
                            
                        except Exception as e:
                            span.set_status(EventStatus.ERROR)
                            span.set_error(e)
                            raise
                
                return sync_wrapper
        return decorator

    def trace_tool_execution_decorator(self, tool_name: str, **kwargs):
        """Decorator for automatically tracing tool executions"""
        def decorator(func: Callable):
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
                            
                            # Set output if result is serializable
                            if isinstance(result, (str, int, float, bool)):
                                span.set_output(str(result)[:1000])
                            elif hasattr(result, '__dict__'):
                                span.set_output(str(result)[:1000])
                            
                            return result
                            
                        except Exception as e:
                            span.set_status(EventStatus.ERROR)
                            span.set_error(e)
                            raise
                
                return async_wrapper
            else:
                @wraps(func)
                def sync_wrapper(*args, **func_kwargs):
                    source_component = kwargs.get('source_component', tool_name)
                    
                    with self.trace_tool_execution_sync(
                        tool_name=tool_name,
                        source_component=source_component,
                        **kwargs
                    ) as span:
                        try:
                            result = func(*args, **func_kwargs)
                            
                            # Set output if result is serializable
                            if isinstance(result, (str, int, float, bool)):
                                span.set_output(str(result)[:1000])
                            
                            return result
                            
                        except Exception as e:
                            span.set_status(EventStatus.ERROR)
                            span.set_error(e)
                            raise
                
                return sync_wrapper
        return decorator

    def trace_agent_action_decorator(self, action_type: str, **kwargs):
        """Decorator for automatically tracing agent actions"""
        def decorator(func: Callable):
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
                            
                            # Set output if result is serializable
                            if isinstance(result, (str, int, float, bool)):
                                span.set_output(str(result)[:1000])
                            
                            return result
                            
                        except Exception as e:
                            span.set_status(EventStatus.ERROR)
                            span.set_error(e)
                            raise
                
                return async_wrapper
            else:
                @wraps(func)
                def sync_wrapper(*args, **func_kwargs):
                    source_component = kwargs.get('source_component', 'agent')
                    
                    with self.trace_agent_action_sync(
                        action_type=action_type,
                        source_component=source_component,
                        **kwargs
                    ) as span:
                        try:
                            result = func(*args, **func_kwargs)
                            
                            # Set output if result is serializable
                            if isinstance(result, (str, int, float, bool)):
                                span.set_output(str(result)[:1000])
                            
                            return result
                            
                        except Exception as e:
                            span.set_status(EventStatus.ERROR)
                            span.set_error(e)
                            raise
                
                return sync_wrapper
        return decorator

    # Utility Methods
    async def health_check(self) -> bool:
        """Check if the telemetry service is healthy"""
        try:
            session = await self._get_session()
            url = f"{self.config.endpoint}/health"
            
            async with session.get(url) as response:
                return response.status == 200
                
        except Exception:
            return False

    async def flush(self) -> None:
        """Flush any pending events"""
        if self._auto_batch_manager:
            await self._auto_batch_manager.flush()

    def get_pending_events_count(self) -> int:
        """Get the number of pending events in auto-batch"""
        if self._auto_batch_manager:
            return self._auto_batch_manager.get_pending_count()
        return 0

    async def close(self) -> None:
        """Close the client and cleanup resources"""
        self._shutdown = True
        
        # Flush any pending events
        await self.flush()
        
        # Close HTTP session
        if self._session and not self._session.closed:
            await self._session.close()
        
        # Wait for background thread to finish
        if self._background_task and self._background_task.is_alive():
            self._background_task.join(timeout=2.0)

    def __del__(self):
        """Cleanup on deletion"""
        if not self._shutdown:
            # Try to schedule cleanup
            try:
                if self._session and not self._session.closed:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        loop.create_task(self._session.close())
            except:
                pass

    # Context manager support for client itself
    async def __aenter__(self) -> 'TelemetryClient':
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()

    def __enter__(self) -> 'TelemetryClient':
        """Sync context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Sync context manager exit"""
        # Schedule async cleanup
        try:
            asyncio.create_task(self.close())
        except:
            self._shutdown = True