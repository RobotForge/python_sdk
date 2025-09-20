
import asyncio
import logging
from client.telemetry_client import TelemetryClient


class FrameworkIntegrations:
    """Higher-level framework integrations"""
    
    def __init__(self, client: TelemetryClient):
        self.client = client

    def wrap_fastapi_app(self, app):
        """Add telemetry middleware to FastAPI app"""
        try:
            from fastapi import Request, Response
            import time
            
            @app.middleware("http")
            async def telemetry_middleware(request: Request, call_next):
                start_time = time.time()
                
                async with self.client.trace_agent_action(
                    action_type="api_request",
                    source_component="fastapi_endpoint"
                ) as span:
                    span.set_input(f"{request.method} {request.url.path}")
                    span.set_metadata("method", request.method)
                    span.set_metadata("path", request.url.path)
                    span.set_metadata("client_ip", request.client.host if request.client else None)
                    
                    try:
                        response = await call_next(request)
                        span.set_metadata("status_code", response.status_code)
                        return response
                    except Exception as e:
                        span.set_metadata("error", str(e))
                        raise
                        
        except ImportError:
            logging.warning("FastAPI not available for instrumentation")

    def wrap_langchain_chain(self, chain):
        """Wrap a LangChain chain with telemetry"""
        try:
            original_call = chain.__call__
            original_acall = getattr(chain, 'acall', None)
            
            async def traced_acall(*args, **kwargs):
                async with self.client.trace_agent_action(
                    action_type="chain_execution",
                    source_component=f"langchain_{chain.__class__.__name__}"
                ) as span:
                    input_data = str(args[0]) if args else str(kwargs)
                    span.set_input(input_data[:1000])
                    
                    try:
                        if original_acall:
                            result = await original_acall(*args, **kwargs)
                        else:
                            # Fallback to sync call in thread
                            import concurrent.futures
                            with concurrent.futures.ThreadPoolExecutor() as executor:
                                result = await asyncio.get_event_loop().run_in_executor(
                                    executor, lambda: original_call(*args, **kwargs)
                                )
                        
                        span.set_output(str(result)[:1000])
                        return result
                    except Exception as e:
                        span.set_metadata("error", str(e))
                        raise
            
            def traced_call(*args, **kwargs):
                # For sync calls, we'd need a different approach
                return original_call(*args, **kwargs)
            
            chain.__call__ = traced_call
            if original_acall:
                chain.acall = traced_acall
                
            return chain
            
        except Exception as e:
            logging.error(f"Failed to wrap LangChain chain: {e}")
            return chain

    def wrap_llamaindex_query_engine(self, query_engine):
        """Wrap a LlamaIndex query engine with telemetry"""
        try:
            original_query = query_engine.query
            original_aquery = getattr(query_engine, 'aquery', None)
            
            async def traced_aquery(query_str, **kwargs):
                async with self.client.trace_agent_action(
                    action_type="index_query",
                    source_component=f"llamaindex_{query_engine.__class__.__name__}"
                ) as span:
                    span.set_input(str(query_str)[:1000])
                    
                    try:
                        if original_aquery:
                            result = await original_aquery(query_str, **kwargs)
                        else:
                            # Fallback to sync query
                            import concurrent.futures
                            with concurrent.futures.ThreadPoolExecutor() as executor:
                                result = await asyncio.get_event_loop().run_in_executor(
                                    executor, lambda: original_query(query_str, **kwargs)
                                )
                        
                        if hasattr(result, 'response'):
                            span.set_output(str(result.response)[:1000])
                        
                        return result
                    except Exception as e:
                        span.set_metadata("error", str(e))
                        raise
            
            def traced_query(query_str, **kwargs):
                return original_query(query_str, **kwargs)
            
            query_engine.query = traced_query
            if original_aquery:
                query_engine.aquery = traced_aquery
                
            return query_engine
            
        except Exception as e:
            logging.error(f"Failed to wrap LlamaIndex query engine: {e}")
            return query_engine