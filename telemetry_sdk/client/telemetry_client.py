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
import asyncio
from concurrent.futures import TimeoutError as FutureTimeoutError
from telemetry_sdk.client.observation_span import ModelCallSpan
from telemetry_sdk.client.pricing import LLMPricing

from .models import (
    TelemetryConfig, TelemetryEvent, EventType, EventStatus, 
    EventIngestionRequest, BatchEventIngestionRequest, APIResponse
)
from .event_builder import EventBuilder, ModelCallEventBuilder, ToolExecutionEventBuilder, AgentActionEventBuilder
from .trace_context import TraceContext, SyncTraceContext
from .batch_manager import BatchManager, AutoBatchManager
from ..utils.exceptions import (
    TelemetrySDKError, NetworkError, AuthenticationError, 
    TimeoutError, PayloadTooLargeError, RateLimitError, BatchError
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
        #endpoint: Optional[str] = None,
        #project_id: Optional[str] = None,
        config: Optional[TelemetryConfig] = None,
        **kwargs
    ):
        # Initialize configuration
        if config:
            self.config = config
        else:
            config_params = {
                'api_key': api_key,
                'endpoint': 'https://cloud.robotforge.com.ng', 
                #'project_id': project_id,
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
        """
        Auto setup background batching using the optimal mode:
        - If running inside an asyncio app: use current event loop
        - If in a sync context: fallback to background thread
        """
        self._auto_batch_manager = AutoBatchManager(self)

        async def background_processor():
            """Continuously process events until shutdown."""
            while not getattr(self, "_shutdown", False):
                try:
                    await self._process_background_events()
                    await asyncio.sleep(0.1)
                except Exception as e:
                    if hasattr(self, "_logger"):
                        self._logger.warning(f"[Telemetry] Background processor error: {e}")
                    await asyncio.sleep(1.0)

        try:
            # ✅ CASE 1: We're in an async environment — run in current loop
            loop = asyncio.get_running_loop()
            loop.create_task(background_processor())
            self._logger.info("[Telemetry] AutoBatchManager started in main event loop")
            self._background_task = None

        except RuntimeError:
            # ✅ CASE 2: No running loop — fallback to background thread
            import threading

            def run_in_thread():
                try:
                    asyncio.run(background_processor())
                except Exception as e:
                    if hasattr(self, "_logger"):
                        self._logger.error(f"[Telemetry] Threaded background processor stopped: {e}")

            thread = threading.Thread(target=run_in_thread, daemon=True)
            thread.start()
            self._background_task = thread
            self._logger.info("[Telemetry] AutoBatchManager started in background thread (sync mode)")



    async def _process_background_events(self):
        """Simplified background loop that works with the new lock-free AutoBatchManager"""
        while not self._shutdown:
            try:
                # Process sync events from threading queue
                events_processed = 0
                
                # Move events from threading queue to async processing
                while not self._sync_event_queue.empty() and events_processed < 50:
                    try:
                        event_data = self._sync_event_queue.get_nowait()
                        event = event_data['event']
                        details = event_data['details']
                        
                        # The new AutoBatchManager handles all the complex logic
                        if self._auto_batch_manager:
                            await self._auto_batch_manager.add_event(event, details)
                        
                        events_processed += 1
                        
                    except Exception as e:
                        # Log but don't break on individual event errors
                        self._logger.debug(f"Error processing sync event: {e}")
                        continue
                
                # Periodic flush check - AutoBatchManager handles the timing logic
                if self._auto_batch_manager:
                    await self._auto_batch_manager.flush()
                
                # Adaptive sleep based on activity
                if events_processed > 0:
                    await asyncio.sleep(0.01)  # 10ms when busy
                else:
                    await asyncio.sleep(0.1)   # 100ms when idle
                    
            except Exception as e:
                self._logger.error(f"Background processing error: {e}")
                await asyncio.sleep(1.0)  # Longer sleep on error to prevent error loops


    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session with proper configuration"""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.config.request_timeout)
            connector = aiohttp.TCPConnector(
                verify_ssl=False,
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
        """Send a batch of telemetry events."""
        url = f"{self.config.endpoint}/api/v1/events/batch"
        payload = request.to_dict()

        try:
            return await self._make_request("POST", url, payload)
        except Exception as e:
            if hasattr(self, "_logger"):
                self._logger.warning(f"Batch send failed: {e}")
            raise BatchError(f"Failed to send batch: {e}") from e


        
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
        



    def _queue_sync_event(self, event, details):
        """
        Queue a telemetry event from a synchronous context.
        Safe to call even when the main loop is running.
        """
        try:
            if not self._auto_batch_manager:
                self._logger.warning("[Telemetry] AutoBatchManager not initialized.")
                return

            loop = None
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                pass  # No running loop (sync context)

            if loop and loop.is_running():
                # Schedule async add_event without blocking
                asyncio.create_task(self._auto_batch_manager.add_event(event, details))
                print(f"[AutoBatch] Queued sync event (async loop) → {event.event_id}")
            else:
                # Run in a temporary loop if not already inside async context
                asyncio.run(self._auto_batch_manager.add_event(event, details))
                print(f"[AutoBatch] Queued sync event (new loop) → {event.event_id}")

        except Exception as e:
            if hasattr(self, "_logger"):
                self._logger.warning(f"[Telemetry] Failed to queue sync event: {e}")

    async def send_event(
        self,
        event: TelemetryEvent,
        details: Optional[Dict[str, Any]] = None,
        *,
        immediate: bool = False
    ) -> APIResponse:
        """
        Public convenience wrapper for sending a single telemetry event.
        
        This method abstracts away batching vs direct send logic, so callers
        (e.g., TraceContext) do not need to touch private methods like
        `_send_event_request`.

        Args:
            event: The TelemetryEvent object to send.
            details: Optional dictionary of additional event details.
            immediate: If True, bypass auto-batching and send immediately.

        Returns:
            APIResponse indicating success or failure.
        """
        if event is None:
            raise ValueError("event cannot be None")

        # Default to empty dict if not provided
        details = details or {}

        # Build ingestion request
        request = EventIngestionRequest(event=event, details=details)

        # If immediate send requested, bypass batching
        if immediate:
            return await self._send_single_event(request)

        # Otherwise, use the client's normal sending logic (auto-batching or immediate)
        return await self._send_event_request(request)


    # Context Manager Methods
    @asynccontextmanager
    async def trace_model_call(self, **kwargs) -> AsyncGenerator[EventBuilder, None]:
        """Context manager for tracing model calls"""
        async with TraceContext(self, EventType.MODEL_CALL, kwargs.pop('source_component', 'model_call'), **kwargs) as builder:
            yield builder

    @asynccontextmanager
    async def trace_tool_execution(self, tool_name: str, **kwargs) -> AsyncGenerator[EventBuilder, None]:
        """Context manager for tracing tool executions"""
        kwargs['tool_name'] = tool_name
        source_component = kwargs.pop('source_component', tool_name)
        async with TraceContext(self, EventType.TOOL_EXECUTION, source_component, **kwargs) as builder:
            yield builder

    @asynccontextmanager
    async def trace_agent_action(self, action_type: str, **kwargs) -> AsyncGenerator[EventBuilder, None]:
        """Context manager for tracing agent actions"""
        kwargs['action_type'] = action_type
        source_component = kwargs.pop('source_component', 'agent')
        async with TraceContext(self, EventType.AGENT_ACTION, source_component, **kwargs) as builder:
            yield builder

    @asynccontextmanager
    async def trace_mcp_event(self, **kwargs) -> AsyncGenerator[EventBuilder, None]:
        """Context manager for tracing MCP events"""
        source_component = kwargs.pop('source_component', 'mcp')
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
        print(source_component)
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

    
    def trace_model_call_decorator(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        *,
        capture_input: bool = True,
        capture_output: bool = True,
        input_serializer: Optional[Callable] = None,
        output_serializer: Optional[Callable] = None,
        **kwargs
    ):
        """
        Enhanced decorator for automatically tracing model calls with input/output capture.
        
        Args:
            provider: LLM provider (e.g., 'openai', 'anthropic')
            model: Model name (e.g., 'gpt-4', 'claude-3-opus')
            capture_input: Whether to automatically capture function input (default: True)
            capture_output: Whether to automatically capture function output (default: True)
            input_serializer: Custom function to serialize input (default: auto-detect)
            output_serializer: Custom function to serialize output (default: auto-detect)
            **kwargs: Additional parameters passed to trace_model_call
        
        Example:
            ```python
            @client.trace_model_call_decorator(
                provider="openai",
                model="gpt-3.5-turbo"
            )
            def get_completion(prompt, model="gpt-3.5-turbo"):
                response = openai.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.choices[0].message.content
            
            # Input (prompt) and output automatically captured
            result = get_completion("What is AI?")
            ```
        """
        def decorator(func: Callable) -> Callable:
            if asyncio.iscoroutinefunction(func):
                @wraps(func)
                async def async_wrapper(*args, **func_kwargs):
                    # Create span with provider/model
                    async with self.trace_model_call(
                        provider=provider,
                        model=model,
                        **kwargs
                    ) as span:
                        try:
                            # ✅ CAPTURE INPUT from function arguments
                            if capture_input:
                                input_data = self._extract_input_from_args(
                                    args, func_kwargs, input_serializer
                                )
                                if input_data is not None:
                                    span.update(input=input_data)
                            
                            # Execute the function
                            result = await func(*args, **func_kwargs)
                            
                            # ✅ CAPTURE OUTPUT and extract metadata from result
                            if capture_output:
                                self._process_llm_result(
                                    span, result, provider, model, output_serializer
                                )
                            
                            return result
                            
                        except Exception as e:
                            span.set_status(EventStatus.ERROR)
                            span.set_error(e)
                            raise
                
                return async_wrapper
            else:
                @wraps(func)
                def sync_wrapper(*args, **func_kwargs):
                    with self.trace_model_call_sync(
                        provider=provider,
                        model=model,
                        **kwargs
                    ) as span:
                        try:
                            # ✅ CAPTURE INPUT
                            if capture_input:
                                input_data = self._extract_input_from_args(
                                    args, func_kwargs, input_serializer
                                )
                                if input_data is not None:
                                    span.update(input=input_data)
                            
                            # Execute the function
                            result = func(*args, **func_kwargs)
                            
                            # ✅ CAPTURE OUTPUT and metadata
                            if capture_output:
                                self._process_llm_result(
                                    span, result, provider, model, output_serializer
                                )
                            
                            return result
                            
                        except Exception as e:
                            span.set_status(EventStatus.ERROR)
                            span.set_error(e)
                            raise
                
                return sync_wrapper
        
        return decorator
    
    def _extract_input_from_args(
        self,
        args: tuple,
        kwargs: dict,
        serializer: Optional[Callable] = None
    ) -> Optional[Any]:
        """
        Extract input data from function arguments.
        Tries to intelligently find the prompt/input parameter.
        """
        if serializer:
            return serializer(args, kwargs)
        
        # Strategy 1: Check for common parameter names in kwargs
        for param_name in ['prompt', 'input', 'query', 'question', 'messages', 'text']:
            if param_name in kwargs:
                return kwargs[param_name]
        
        # Strategy 2: Use first positional argument if available
        if args:
            return args[0]
        
        # Strategy 3: Return first kwarg value if only one exists
        if len(kwargs) == 1:
            return list(kwargs.values())[0]
        
        return None
    
    def _process_llm_result(
        self,
        span: ModelCallSpan,
        result: Any,
        provider: Optional[str],
        model: Optional[str],
        serializer: Optional[Callable] = None
    ) -> None:
        """
        Process LLM result to extract output, tokens, cost, etc.
        Handles common LLM response formats from OpenAI, Anthropic, etc.
        """
        if serializer:
            output = serializer(result)
            span.update(output=output)
            return
        
        # Handle string results (already extracted content)
        if isinstance(result, str):
            span.update(output=result)
            return
        
        # Handle OpenAI-style responses
        if hasattr(result, 'choices') and result.choices:
            choice = result.choices[0]
            
            # Extract output text
            if hasattr(choice, 'message'):
                output_text = choice.message.content
                span.update(output=output_text)
            elif hasattr(choice, 'text'):
                output_text = choice.text
                span.update(output=output_text)
            
            # Extract finish reason
            if hasattr(choice, 'finish_reason'):
                span.set_finish_reason(choice.finish_reason)
        
        # Handle usage/token information
        if hasattr(result, 'usage'):
            usage = result.usage
            usage_dict = {
                'prompt_tokens': getattr(usage, 'prompt_tokens', 0),
                'completion_tokens': getattr(usage, 'completion_tokens', 0),
                'total_tokens': getattr(usage, 'total_tokens', 0)
            }
            
            # Set usage details
            span.set_usage_details(usage_dict)
            
            # Calculate and set cost if provider and model are known
            if provider and model and usage_dict['total_tokens'] > 0:
                cost = LLMPricing.calculate_cost(
                    provider=provider,
                    model=model,
                    prompt_tokens=usage_dict['prompt_tokens'],
                    completion_tokens=usage_dict['completion_tokens']
                )
                if cost is not None:
                    span.set_cost_details(cost)
        
        # Handle Anthropic-style responses
        elif hasattr(result, 'content') and hasattr(result, 'usage'):
            # Extract content
            if isinstance(result.content, list) and len(result.content) > 0:
                output_text = result.content[0].text if hasattr(result.content[0], 'text') else str(result.content[0])
                span.update(output=output_text)
            elif isinstance(result.content, str):
                span.update(output=result.content)
            
            # Extract usage
            if hasattr(result.usage, 'input_tokens'):
                usage_dict = {
                    'prompt_tokens': result.usage.input_tokens,
                    'completion_tokens': result.usage.output_tokens,
                    'total_tokens': result.usage.input_tokens + result.usage.output_tokens
                }
                span.set_usage_details(usage_dict)
                
                if provider and model:
                    cost = LLMPricing.calculate_cost(
                        provider=provider,
                        model=model,
                        prompt_tokens=usage_dict['prompt_tokens'],
                        completion_tokens=usage_dict['completion_tokens']
                    )
                    if cost is not None:
                        span.set_cost_details(cost)
            
            # Extract stop reason
            if hasattr(result, 'stop_reason'):
                span.set_finish_reason(result.stop_reason)
    
    def trace_tool_execution_decorator(
        self,
        tool_name: str,
        *,
        capture_input: bool = True,
        capture_output: bool = True,
        **kwargs
    ):
        """
        Enhanced decorator for automatically tracing tool executions.
        
        Args:
            tool_name: Name of the tool being executed
            capture_input: Whether to automatically capture function input (default: True)
            capture_output: Whether to automatically capture function output (default: True)
            **kwargs: Additional parameters passed to trace_tool_execution
        
        Example:
            ```python
            @client.trace_tool_execution_decorator(tool_name="web_search")
            def search_web(query: str) -> list:
                results = requests.get(f"https://api.search.com?q={query}")
                return results.json()
            
            # Input and output automatically captured
            results = search_web("AI news")
            ```
        """
        def decorator(func: Callable) -> Callable:
            if asyncio.iscoroutinefunction(func):
                @wraps(func)
                async def async_wrapper(*args, **func_kwargs):
                    async with self.trace_tool_execution(
                        tool_name=tool_name,
                        **kwargs
                    ) as span:
                        try:
                            # Capture input
                            if capture_input:
                                input_data = self._extract_input_from_args(args, func_kwargs)
                                if input_data is not None:
                                    span.update(input=input_data)
                            
                            # Execute function
                            result = await func(*args, **func_kwargs)
                            
                            # Capture output
                            if capture_output and result is not None:
                                span.update(output=result)
                            
                            return result
                            
                        except Exception as e:
                            span.set_status(EventStatus.ERROR)
                            span.set_error(e)
                            raise
                
                return async_wrapper
            else:
                @wraps(func)
                def sync_wrapper(*args, **func_kwargs):
                    with self.trace_tool_execution_sync(
                        tool_name=tool_name,
                        **kwargs
                    ) as span:
                        try:
                            # Capture input
                            if capture_input:
                                input_data = self._extract_input_from_args(args, func_kwargs)
                                if input_data is not None:
                                    span.update(input=input_data)
                            
                            # Execute function
                            result = func(*args, **func_kwargs)
                            
                            # Capture output
                            if capture_output and result is not None:
                                span.update(output=result)
                            
                            return result
                            
                        except Exception as e:
                            span.set_status(EventStatus.ERROR)
                            span.set_error(e)
                            raise
                
                return sync_wrapper
        
        return decorator
    
    def trace_agent_action_decorator(
        self,
        action_type: str,
        *,
        capture_input: bool = True,
        capture_output: bool = True,
        **kwargs
    ):
        """
        Enhanced decorator for automatically tracing agent actions.
        
        Args:
            action_type: Type of agent action (e.g., 'planning', 'reasoning', 'decision')
            capture_input: Whether to automatically capture function input (default: True)
            capture_output: Whether to automatically capture function output (default: True)
            **kwargs: Additional parameters passed to trace_agent_action
        
        Example:
            ```python
            @client.trace_agent_action_decorator(action_type="planning")
            def plan_tasks(goal: str) -> list:
                # Agent planning logic
                tasks = break_down_goal(goal)
                return tasks
            
            # Input and output automatically captured
            tasks = plan_tasks("Build a web app")
            ```
        """
        def decorator(func: Callable) -> Callable:
            if asyncio.iscoroutinefunction(func):
                @wraps(func)
                async def async_wrapper(*args, **func_kwargs):
                    async with self.trace_agent_action(
                        action_type=action_type,
                        **kwargs
                    ) as span:
                        try:
                            # Capture input
                            if capture_input:
                                input_data = self._extract_input_from_args(args, func_kwargs)
                                if input_data is not None:
                                    span.update(input=input_data)
                            
                            # Execute function
                            result = await func(*args, **func_kwargs)
                            
                            # Capture output
                            if capture_output and result is not None:
                                span.update(output=result)
                            
                            return result
                            
                        except Exception as e:
                            span.set_status(EventStatus.ERROR)
                            span.set_error(e)
                            raise
                
                return async_wrapper
            else:
                @wraps(func)
                def sync_wrapper(*args, **func_kwargs):
                    with self.trace_agent_action_sync(
                        action_type=action_type,
                        **kwargs
                    ) as span:
                        try:
                            # Capture input
                            if capture_input:
                                input_data = self._extract_input_from_args(args, func_kwargs)
                                if input_data is not None:
                                    span.update(input=input_data)
                            
                            # Execute function
                            result = func(*args, **func_kwargs)
                            
                            # Capture output
                            if capture_output and result is not None:
                                span.update(output=result)
                            
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



    async def _flush_async(self) -> None:
        """Internal async flush logic — processes sync + async queues."""
        # First, handle sync-queued events if any
        if hasattr(self, "_process_background_events_once"):
            await self._process_background_events_once()

        # Then flush async auto-batch manager
        if self._auto_batch_manager:
            await self._auto_batch_manager.flush()


    def flush(self) -> None:
        """
        Unified flush — safe to call from either sync or async context.
        Automatically detects the event loop state.
        """
        async def _flush():
            await self._flush_async()

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # ✅ Already inside event loop (async context)
            loop.create_task(_flush())
        else:
            # ✅ No active loop (sync context)
            asyncio.run(_flush())





    def get_pending_events_count(self) -> int:
        """Get the number of pending events in auto-batch"""
        if self._auto_batch_manager:
            return self._auto_batch_manager.get_pending_count()
        return 0


    async def _process_sync_queue_into_autobatch(self, max_items: int = 500) -> int:
        """
        Move items from the sync threading queue (_sync_event_queue) into the AutoBatchManager.
        Returns number of events moved.
        Must be called inside an event loop.
        """
        moved = 0
        # Safety: limit how many we move in one pass
        while not self._sync_event_queue.empty() and moved < max_items:
            try:
                event_data = self._sync_event_queue.get_nowait()
            except Exception:
                break
            try:
                event = event_data["event"]
                details = event_data.get("details", {}) or {}
                # Use auto_batch_manager.add_event (awaitable)
                if self._auto_batch_manager:
                    await self._auto_batch_manager.add_event(event, details)
                moved += 1
            except Exception as e:
                # Log and continue
                if hasattr(self, "_logger"):
                    self._logger.debug(f"[Telemetry] Error moving sync event to autobatch: {e}")
                continue
        return moved


    def flush(self, timeout: float = 5.0) -> None:
        """
        Unified, loop-safe flush.
        - If called from an async context, schedules and awaits flushing synchronously.
        - If called from a sync context, runs a temporary loop to perform the flush.
        This ensures the sync->async handoff completes and the auto-batch manager flushes.
        """
        async def _flush_coroutine():
            # 1) If we are inside an event loop, first process the sync queue into the autobatcher
            #    (this ensures events created in sync contexts get moved).
            try:
                # If we have a _sync_event_queue, move its items into the autobatch
                if hasattr(self, "_sync_event_queue") and not self._sync_event_queue.empty():
                    # This helper uses await self._auto_batch_manager.add_event internally
                    await self._process_sync_queue_into_autobatch()
            except Exception as e:
                if hasattr(self, "_logger"):
                    self._logger.debug(f"[Telemetry] Error while processing sync queue: {e}")

            # 2) Trigger a flush on the AutoBatchManager (which may run on a background loop)
            if self._auto_batch_manager:
                # If auto-batch manager has a dedicated background loop, use that loop to flush
                bg_loop = getattr(self._auto_batch_manager, "_background_loop", None)
                try:
                    # If we're in the same loop that bg_loop references, just await
                    running_loop = None
                    try:
                        running_loop = asyncio.get_running_loop()
                    except RuntimeError:
                        running_loop = None

                    if bg_loop is None or bg_loop == running_loop:
                        # Await directly on the manager's flush
                        await self._auto_batch_manager.flush()
                    else:
                        # We must schedule the flush to run in the background loop and wait for it
                        fut = asyncio.run_coroutine_threadsafe(self._auto_batch_manager.flush(), bg_loop)
                        # Wait (blocking) for result up to timeout
                        try:
                            await asyncio.wrap_future(fut)  # convert to awaitable and await it
                        except FutureTimeoutError:
                            # cancel future if possible
                            try:
                                fut.cancel()
                            except Exception:
                                pass
                            raise
                except Exception as e:
                    # Bubble up a nice log but don't crash the app
                    if hasattr(self, "_logger"):
                        self._logger.warning(f"[Telemetry] Flush failed: {e}")
            else:
                # no auto_batch_manager; nothing else to do
                return

        # Determine whether we're in an active running loop
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # We're inside async context — create a Task and await it to completion
            # NOTE: we await here by scheduling and awaiting directly (caller likely will await)
            # but because flush() is synchronous API we schedule task and wait synchronously using run_coroutine_threadsafe
            # to avoid nested-run issues. Instead, we simply create a task and return it to caller.
            # To keep the external API simple, we schedule and run it via run_coroutine_threadsafe on this loop.
            # However run_coroutine_threadsafe requires a loop different from the current — so just create a task and return.
            task = loop.create_task(_flush_coroutine())
            # Caller in async context probably expects to await flush — so return the task to be awaited.
            # But because this flush() is synchronous API, we leave it non-blocking: schedule task and return.
            return task
        else:
            # No running loop — run the coroutine to completion in a new loop (blocking)
            asyncio.run(_flush_coroutine())
            return None
        



    async def close(self) -> None:
        """
        Gracefully close telemetry client and cleanup resources.
        Ensures pending events are flushed before shutdown,
        even if running in mixed sync/async or multi-threaded contexts.
        """
        self._shutdown = True

        try:
            # ✅ Ensure flush completes before loop closes
            if hasattr(self, "_auto_batch_manager") and self._auto_batch_manager:
                await self._auto_batch_manager.flush()
                await asyncio.sleep(0.05)
                await self._auto_batch_manager.flush()
            else:
                await self.flush()
        except RuntimeError as e:
            if "Event loop is closed" in str(e):
                # 🔁 Create a temporary loop to finish flushing
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    new_loop.run_until_complete(self.flush())
                finally:
                    new_loop.close()
            else:
                self._logger.warning(f"[Telemetry] Flush during close failed: {e}")
        except Exception as e:
            self._logger.warning(f"[Telemetry] Flush during close failed: {e}")

        # ✅ Close aiohttp session safely
        try:
            if hasattr(self, "_session") and self._session and not self._session.closed:
                await self._session.close()
        except RuntimeError as e:
            if "attached to a different loop" in str(e):
                loop = asyncio.get_event_loop()
                fut = asyncio.run_coroutine_threadsafe(self._session.close(), loop)
                await asyncio.wrap_future(fut)
            else:
                raise
        except Exception as e:
            self._logger.warning(f"[Telemetry] Session close error: {e}")

        # ✅ Join background thread if it exists
        if getattr(self, "_background_task", None):
            try:
                self._background_task.join(timeout=2.0)
                self._logger.info("[Telemetry] Background thread joined successfully")
            except Exception as e:
                self._logger.warning(f"[Telemetry] Background thread join failed: {e}")


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