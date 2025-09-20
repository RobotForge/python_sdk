"""
Framework-specific integrations for the Telemetry SDK
"""

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from ..client.telemetry_client import TelemetryClient


class FrameworkIntegrations:
    """High-level framework integrations for popular Python frameworks"""
    
    def __init__(self, client: 'TelemetryClient'):
        self.client = client
        self._logger = logging.getLogger(__name__)

    def wrap_fastapi_app(self, app) -> None:
        """Add telemetry middleware to FastAPI app"""
        try:
            from fastapi import Request, Response
            import time
            
            @app.middleware("http")
            async def telemetry_middleware(request: Request, call_next):
                start_time = time.time()
                
                async with self.client.trace_agent_action(
                    action_type="api_request",
                    source_component="fastapi_middleware"
                ) as span:
                    # Set request information
                    span.set_input(f"{request.method} {request.url.path}")
                    span.set_metadata("method", request.method)
                    span.set_metadata("path", request.url.path)
                    span.set_metadata("query_params", str(request.query_params))
                    
                    if request.client:
                        span.set_metadata("client_ip", request.client.host)
                    
                    # Add user agent if available
                    user_agent = request.headers.get("user-agent")
                    if user_agent:
                        span.set_metadata("user_agent", user_agent[:200])
                    
                    try:
                        response = await call_next(request)
                        
                        # Set response information
                        span.set_metadata("status_code", response.status_code)
                        span.set_metadata("response_time_ms", int((time.time() - start_time) * 1000))
                        
                        return response
                        
                    except Exception as e:
                        span.set_metadata("error", str(e))
                        span.set_metadata("response_time_ms", int((time.time() - start_time) * 1000))
                        raise
                        
            self._logger.info("FastAPI telemetry middleware installed")
            
        except ImportError:
            self._logger.warning("FastAPI not available for instrumentation")
        except Exception as e:
            self._logger.error(f"Failed to install FastAPI middleware: {e}")

    def wrap_flask_app(self, app) -> None:
        """Add telemetry to Flask app"""
        try:
            from flask import request, g
            import time
            
            @app.before_request
            def before_request():
                g.telemetry_start_time = time.time()
                g.telemetry_span = self.client.trace_agent_action_sync(
                    action_type="api_request",
                    source_component="flask_middleware"
                ).__enter__()
                
                # Set request information
                g.telemetry_span.set_input(f"{request.method} {request.path}")
                g.telemetry_span.set_metadata("method", request.method)
                g.telemetry_span.set_metadata("path", request.path)
                g.telemetry_span.set_metadata("remote_addr", request.remote_addr)
                
                if request.user_agent:
                    g.telemetry_span.set_metadata("user_agent", str(request.user_agent)[:200])
            
            @app.after_request
            def after_request(response):
                if hasattr(g, 'telemetry_span') and hasattr(g, 'telemetry_start_time'):
                    # Set response information
                    g.telemetry_span.set_metadata("status_code", response.status_code)
                    response_time = int((time.time() - g.telemetry_start_time) * 1000)
                    g.telemetry_span.set_metadata("response_time_ms", response_time)
                    
                    # Exit the span context
                    g.telemetry_span.__exit__(None, None, None)
                
                return response
            
            @app.errorhandler(Exception)
            def handle_exception(e):
                if hasattr(g, 'telemetry_span'):
                    g.telemetry_span.set_metadata("error", str(e))
                    g.telemetry_span.__exit__(type(e), e, None)
                raise
            
            self._logger.info("Flask telemetry hooks installed")
            
        except ImportError:
            self._logger.warning("Flask not available for instrumentation")
        except Exception as e:
            self._logger.error(f"Failed to install Flask hooks: {e}")

    def wrap_langchain_chain(self, chain):
        """Wrap a LangChain chain with telemetry"""
        try:
            original_call = getattr(chain, '__call__', None)
            original_acall = getattr(chain, 'acall', None)
            original_invoke = getattr(chain, 'invoke', None)
            original_ainvoke = getattr(chain, 'ainvoke', None)
            
            # Wrap async call method
            if original_acall:
                async def traced_acall(*args, **kwargs):
                    async with self.client.trace_agent_action(
                        action_type="chain_execution",
                        source_component=f"langchain_{chain.__class__.__name__}"
                    ) as span:
                        input_data = str(args[0]) if args else str(kwargs)
                        span.set_input(input_data[:1000])
                        
                        try:
                            result = await original_acall(*args, **kwargs)
                            span.set_output(str(result)[:1000])
                            return result
                        except Exception as e:
                            span.set_metadata("error", str(e))
                            raise
                
                chain.acall = traced_acall
            
            # Wrap sync call method
            if original_call:
                def traced_call(*args, **kwargs):
                    with self.client.trace_agent_action_sync(
                        action_type="chain_execution",
                        source_component=f"langchain_{chain.__class__.__name__}"
                    ) as span:
                        input_data = str(args[0]) if args else str(kwargs)
                        span.set_input(input_data[:1000])
                        
                        try:
                            result = original_call(*args, **kwargs)
                            span.set_output(str(result)[:1000])
                            return result
                        except Exception as e:
                            span.set_metadata("error", str(e))
                            raise
                
                chain.__call__ = traced_call
            
            # Wrap invoke methods (newer LangChain versions)
            if original_ainvoke:
                async def traced_ainvoke(*args, **kwargs):
                    async with self.client.trace_agent_action(
                        action_type="chain_invoke",
                        source_component=f"langchain_{chain.__class__.__name__}"
                    ) as span:
                        input_data = str(args[0]) if args else str(kwargs)
                        span.set_input(input_data[:1000])
                        
                        try:
                            result = await original_ainvoke(*args, **kwargs)
                            span.set_output(str(result)[:1000])
                            return result
                        except Exception as e:
                            span.set_metadata("error", str(e))
                            raise
                
                chain.ainvoke = traced_ainvoke
            
            if original_invoke:
                def traced_invoke(*args, **kwargs):
                    with self.client.trace_agent_action_sync(
                        action_type="chain_invoke",
                        source_component=f"langchain_{chain.__class__.__name__}"
                    ) as span:
                        input_data = str(args[0]) if args else str(kwargs)
                        span.set_input(input_data[:1000])
                        
                        try:
                            result = original_invoke(*args, **kwargs)
                            span.set_output(str(result)[:1000])
                            return result
                        except Exception as e:
                            span.set_metadata("error", str(e))
                            raise
                
                chain.invoke = traced_invoke
            
            self._logger.debug(f"LangChain chain {chain.__class__.__name__} wrapped with telemetry")
            return chain
            
        except Exception as e:
            self._logger.error(f"Failed to wrap LangChain chain: {e}")
            return chain

    def wrap_llamaindex_query_engine(self, query_engine):
        """Wrap a LlamaIndex query engine with telemetry"""
        try:
            original_query = getattr(query_engine, 'query', None)
            original_aquery = getattr(query_engine, 'aquery', None)
            
            # Wrap async query
            if original_aquery:
                async def traced_aquery(query_str, **kwargs):
                    async with self.client.trace_agent_action(
                        action_type="index_query",
                        source_component=f"llamaindex_{query_engine.__class__.__name__}"
                    ) as span:
                        span.set_input(str(query_str)[:1000])
                        
                        try:
                            result = await original_aquery(query_str, **kwargs)
                            
                            if hasattr(result, 'response'):
                                span.set_output(str(result.response)[:1000])
                            
                            if hasattr(result, 'source_nodes'):
                                span.set_metadata("source_nodes_count", len(result.source_nodes))
                            
                            return result
                        except Exception as e:
                            span.set_metadata("error", str(e))
                            raise
                
                query_engine.aquery = traced_aquery
            
            # Wrap sync query
            if original_query:
                def traced_query(query_str, **kwargs):
                    with self.client.trace_agent_action_sync(
                        action_type="index_query",
                        source_component=f"llamaindex_{query_engine.__class__.__name__}"
                    ) as span:
                        span.set_input(str(query_str)[:1000])
                        
                        try:
                            result = original_query(query_str, **kwargs)
                            
                            if hasattr(result, 'response'):
                                span.set_output(str(result.response)[:1000])
                            
                            if hasattr(result, 'source_nodes'):
                                span.set_metadata("source_nodes_count", len(result.source_nodes))
                            
                            return result
                        except Exception as e:
                            span.set_metadata("error", str(e))
                            raise
                
                query_engine.query = traced_query
            
            self._logger.debug(f"LlamaIndex query engine {query_engine.__class__.__name__} wrapped with telemetry")
            return query_engine
            
        except Exception as e:
            self._logger.error(f"Failed to wrap LlamaIndex query engine: {e}")
            return query_engine

    def wrap_llamaindex_chat_engine(self, chat_engine):
        """Wrap a LlamaIndex chat engine with telemetry"""
        try:
            original_chat = getattr(chat_engine, 'chat', None)
            original_achat = getattr(chat_engine, 'achat', None)
            
            # Wrap async chat
            if original_achat:
                async def traced_achat(message, **kwargs):
                    async with self.client.trace_agent_action(
                        action_type="chat",
                        source_component=f"llamaindex_{chat_engine.__class__.__name__}"
                    ) as span:
                        span.set_input(str(message)[:1000])
                        
                        try:
                            result = await original_achat(message, **kwargs)
                            
                            if hasattr(result, 'response'):
                                span.set_output(str(result.response)[:1000])
                            elif isinstance(result, str):
                                span.set_output(result[:1000])
                            
                            return result
                        except Exception as e:
                            span.set_metadata("error", str(e))
                            raise
                
                chat_engine.achat = traced_achat
            
            # Wrap sync chat
            if original_chat:
                def traced_chat(message, **kwargs):
                    with self.client.trace_agent_action_sync(
                        action_type="chat",
                        source_component=f"llamaindex_{chat_engine.__class__.__name__}"
                    ) as span:
                        span.set_input(str(message)[:1000])
                        
                        try:
                            result = original_chat(message, **kwargs)
                            
                            if hasattr(result, 'response'):
                                span.set_output(str(result.response)[:1000])
                            elif isinstance(result, str):
                                span.set_output(result[:1000])
                            
                            return result
                        except Exception as e:
                            span.set_metadata("error", str(e))
                            raise
                
                chat_engine.chat = traced_chat
            
            self._logger.debug(f"LlamaIndex chat engine {chat_engine.__class__.__name__} wrapped with telemetry")
            return chat_engine
            
        except Exception as e:
            self._logger.error(f"Failed to wrap LlamaIndex chat engine: {e}")
            return chat_engine

    def wrap_django_view(self, view_func):
        """Wrap a Django view function with telemetry"""
        try:
            from functools import wraps
            
            @wraps(view_func)
            def wrapped_view(request, *args, **kwargs):
                with self.client.trace_agent_action_sync(
                    action_type="django_view",
                    source_component="django_view"
                ) as span:
                    # Set request information
                    span.set_input(f"{request.method} {request.path}")
                    span.set_metadata("method", request.method)
                    span.set_metadata("path", request.path)
                    
                    if hasattr(request, 'user') and request.user:
                        span.set_metadata("user_authenticated", request.user.is_authenticated)
                    
                    try:
                        response = view_func(request, *args, **kwargs)
                        
                        # Set response information
                        if hasattr(response, 'status_code'):
                            span.set_metadata("status_code", response.status_code)
                        
                        return response
                        
                    except Exception as e:
                        span.set_metadata("error", str(e))
                        raise
            
            return wrapped_view
            
        except Exception as e:
            self._logger.error(f"Failed to wrap Django view: {e}")
            return view_func

    def create_decorator(self, event_type: str = "agent_action", **default_kwargs):
        """Create a custom decorator for any function"""
        def decorator(func):
            if asyncio.iscoroutinefunction(func):
                async def async_wrapper(*args, **kwargs):
                    if event_type == "model_call":
                        context_manager = self.client.trace_model_call(**default_kwargs)
                    elif event_type == "tool_execution":
                        tool_name = default_kwargs.get('tool_name', func.__name__)
                        context_manager = self.client.trace_tool_execution(tool_name, **default_kwargs)
                    else:
                        action_type = default_kwargs.get('action_type', func.__name__)
                        context_manager = self.client.trace_agent_action(action_type, **default_kwargs)
                    
                    async with context_manager as span:
                        try:
                            result = await func(*args, **kwargs)
                            return result
                        except Exception as e:
                            span.set_metadata("error", str(e))
                            raise
                
                return async_wrapper
            else:
                def sync_wrapper(*args, **kwargs):
                    if event_type == "model_call":
                        context_manager = self.client.trace_model_call_sync(**default_kwargs)
                    elif event_type == "tool_execution":
                        tool_name = default_kwargs.get('tool_name', func.__name__)
                        context_manager = self.client.trace_tool_execution_sync(tool_name, **default_kwargs)
                    else:
                        action_type = default_kwargs.get('action_type', func.__name__)
                        context_manager = self.client.trace_agent_action_sync(action_type, **default_kwargs)
                    
                    with context_manager as span:
                        try:
                            result = func(*args, **kwargs)
                            return result
                        except Exception as e:
                            span.set_metadata("error", str(e))
                            raise
                
                return sync_wrapper
        
        return decorator