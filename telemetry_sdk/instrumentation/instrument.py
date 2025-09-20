"""
Auto-instrumentation module for popular AI/ML libraries
Automatically patches common libraries to capture telemetry without code changes
"""

import functools
import inspect
import logging
from typing import Any, Dict, Optional, Callable, Union
import asyncio
from contextlib import asynccontextmanager

from client.models import EventType
from client.telemetry_client import TelemetryClient




class AutoInstrumentation:
    """Auto-instrumentation manager"""
    
    def __init__(self, client: TelemetryClient):
        self.client = client
        self.original_methods = {}
        self.patched_modules = set()

    def instrument_all(self):
        """Instrument all supported libraries"""
        self.instrument_openai()
        self.instrument_anthropic()
        self.instrument_langchain()
        self.instrument_llamaindex()
        self.instrument_requests()

    def uninstrument_all(self):
        """Remove all instrumentation"""
        for module_name, methods in self.original_methods.items():
            try:
                module = __import__(module_name, fromlist=[''])
                for method_path, original_method in methods.items():
                    self._set_nested_attr(module, method_path, original_method)
            except ImportError:
                continue
        
        self.original_methods.clear()
        self.patched_modules.clear()

    def _patch_method(self, module_name: str, method_path: str, wrapper_func: Callable):
        """Generic method patching utility"""
        try:
            module = __import__(module_name, fromlist=[''])
            original_method = self._get_nested_attr(module, method_path)
            
            if original_method is None:
                return False
            
            # Store original method
            if module_name not in self.original_methods:
                self.original_methods[module_name] = {}
            self.original_methods[module_name][method_path] = original_method
            
            # Apply wrapper
            wrapped_method = wrapper_func(original_method)
            self._set_nested_attr(module, method_path, wrapped_method)
            
            self.patched_modules.add(module_name)
            return True
            
        except ImportError:
            logging.debug(f"Module {module_name} not available for instrumentation")
            return False
        except Exception as e:
            logging.error(f"Failed to patch {module_name}.{method_path}: {e}")
            return False

    def _get_nested_attr(self, obj, path: str):
        """Get nested attribute using dot notation"""
        for attr in path.split('.'):
            obj = getattr(obj, attr, None)
            if obj is None:
                return None
        return obj

    def _set_nested_attr(self, obj, path: str, value):
        """Set nested attribute using dot notation"""
        attrs = path.split('.')
        for attr in attrs[:-1]:
            obj = getattr(obj, attr)
        setattr(obj, attrs[-1], value)

    def instrument_openai(self):
        """Instrument OpenAI client"""
        
        def create_openai_wrapper(original_method):
            if asyncio.iscoroutinefunction(original_method):
                @functools.wraps(original_method)
                async def async_wrapper(*args, **kwargs):
                    # Extract model and other details from args/kwargs
                    model = kwargs.get('model', 'unknown')
                    messages = kwargs.get('messages', [])
                    
                    input_text = ""
                    if messages and isinstance(messages, list):
                        input_text = str(messages[-1].get('content', '')) if messages else ""
                    
                    async with self.client.trace_model_call(
                        provider="openai",
                        model=model,
                        source_component="openai_client"
                    ) as span:
                        span.set_input(input_text[:1000])  # Limit input size
                        
                        try:
                            result = await original_method(*args, **kwargs)
                            
                            # Extract response details
                            if hasattr(result, 'choices') and result.choices:
                                output_text = result.choices[0].message.content if hasattr(result.choices[0], 'message') else ""
                                span.set_output(output_text[:1000])
                            
                            if hasattr(result, 'usage'):
                                span.set_tokens(result.usage.total_tokens)
                                # Estimate cost based on model and tokens
                                cost = self._estimate_openai_cost(model, result.usage.total_tokens)
                                if cost:
                                    span.set_cost(cost)
                            
                            span.set_metadata("finish_reason", getattr(result.choices[0], 'finish_reason', None) if hasattr(result, 'choices') and result.choices else None)
                            
                            return result
                            
                        except Exception as e:
                            span.set_metadata("error", str(e))
                            raise
                            
                return async_wrapper
            else:
                @functools.wraps(original_method)
                def sync_wrapper(*args, **kwargs):
                    # For sync methods, create event manually
                    start_time = asyncio.get_event_loop().time()
                    try:
                        result = original_method(*args, **kwargs)
                        # Create and send event (simplified)
                        return result
                    except Exception as e:
                        # Log error
                        raise
                return sync_wrapper

        # Patch different OpenAI client versions
        patterns = [
            ("openai", "ChatCompletion.acreate"),
            ("openai", "ChatCompletion.create"),
            ("openai", "Completion.acreate"),
            ("openai", "Completion.create"),
            ("openai.resources.chat.completions", "Completions.create"),
            ("openai.resources.completions", "Completions.create"),
        ]
        
        for module_name, method_path in patterns:
            self._patch_method(module_name, method_path, create_openai_wrapper)

    def instrument_anthropic(self):
        """Instrument Anthropic Claude client"""
        
        def create_anthropic_wrapper(original_method):
            if asyncio.iscoroutinefunction(original_method):
                @functools.wraps(original_method)
                async def async_wrapper(*args, **kwargs):
                    model = kwargs.get('model', 'claude')
                    messages = kwargs.get('messages', [])
                    
                    input_text = ""
                    if messages and isinstance(messages, list):
                        input_text = str(messages[-1].get('content', '')) if messages else ""
                    
                    async with self.client.trace_model_call(
                        provider="anthropic",
                        model=model,
                        source_component="anthropic_client"
                    ) as span:
                        span.set_input(input_text[:1000])
                        
                        try:
                            result = await original_method(*args, **kwargs)
                            
                            if hasattr(result, 'content') and result.content:
                                output_text = result.content[0].text if isinstance(result.content, list) else str(result.content)
                                span.set_output(output_text[:1000])
                            
                            if hasattr(result, 'usage'):
                                total_tokens = getattr(result.usage, 'input_tokens', 0) + getattr(result.usage, 'output_tokens', 0)
                                span.set_tokens(total_tokens)
                            
                            return result
                            
                        except Exception as e:
                            span.set_metadata("error", str(e))
                            raise
                            
                return async_wrapper
            else:
                @functools.wraps(original_method)
                def sync_wrapper(*args, **kwargs):
                    return original_method(*args, **kwargs)
                return sync_wrapper

        patterns = [
            ("anthropic", "Anthropic.messages.create"),
            ("anthropic", "AsyncAnthropic.messages.create"),
            ("anthropic.resources.messages", "Messages.create"),
        ]
        
        for module_name, method_path in patterns:
            self._patch_method(module_name, method_path, create_anthropic_wrapper)

    def instrument_langchain(self):
        """Instrument LangChain components"""
        
        def create_langchain_wrapper(original_method):
            if asyncio.iscoroutinefunction(original_method):
                @functools.wraps(original_method)
                async def async_wrapper(self_obj, *args, **kwargs):
                    # Determine component type
                    component_name = self_obj.__class__.__name__
                    
                    if 'LLM' in component_name or 'ChatModel' in component_name:
                        event_type = EventType.MODEL_CALL
                    elif 'Tool' in component_name:
                        event_type = EventType.TOOL_EXECUTION
                    else:
                        event_type = EventType.AGENT_ACTION
                    
                    context_method = {
                        EventType.MODEL_CALL: self.client.trace_model_call,
                        EventType.TOOL_EXECUTION: self.client.trace_tool_execution,
                        EventType.AGENT_ACTION: self.client.trace_agent_action,
                    }[event_type]
                    
                    context_kwargs = {'source_component': f'langchain_{component_name}'}
                    if event_type == EventType.TOOL_EXECUTION:
                        context_kwargs['tool_name'] = component_name
                    elif event_type == EventType.AGENT_ACTION:
                        context_kwargs['action_type'] = 'chain_execution'
                    
                    async with context_method(**context_kwargs) as span:
                        input_data = str(args[0]) if args else str(kwargs)
                        span.set_input(input_data[:1000])
                        
                        try:
                            result = await original_method(self_obj, *args, **kwargs)
                            span.set_output(str(result)[:1000])
                            return result
                        except Exception as e:
                            span.set_metadata("error", str(e))
                            raise
                            
                return async_wrapper
            else:
                @functools.wraps(original_method)
                def sync_wrapper(self_obj, *args, **kwargs):
                    return original_method(self_obj, *args, **kwargs)
                return sync_wrapper

        patterns = [
            ("langchain.llms.base", "BaseLLM._call"),
            ("langchain.llms.base", "BaseLLM._acall"),
            ("langchain.chat_models.base", "BaseChatModel._call"),
            ("langchain.chat_models.base", "BaseChatModel._acall"),
            ("langchain.tools.base", "BaseTool._run"),
            ("langchain.tools.base", "BaseTool._arun"),
        ]
        
        for module_name, method_path in patterns:
            self._patch_method(module_name, method_path, create_langchain_wrapper)

    def instrument_llamaindex(self):
        """Instrument LlamaIndex components"""
        
        def create_llamaindex_wrapper(original_method):
            if asyncio.iscoroutinefunction(original_method):
                @functools.wraps(original_method)
                async def async_wrapper(self_obj, *args, **kwargs):
                    component_name = self_obj.__class__.__name__
                    
                    async with self.client.trace_agent_action(
                        action_type="query_execution",
                        source_component=f'llamaindex_{component_name}'
                    ) as span:
                        # Extract query if available
                        query = args[0] if args and isinstance(args[0], str) else str(kwargs.get('query', ''))
                        span.set_input(query[:1000])
                        
                        try:
                            result = await original_method(self_obj, *args, **kwargs)
                            
                            # Extract response text if available
                            if hasattr(result, 'response'):
                                span.set_output(str(result.response)[:1000])
                            elif isinstance(result, str):
                                span.set_output(result[:1000])
                            
                            # Extract metadata if available
                            if hasattr(result, 'metadata'):
                                span.set_metadata("source_nodes", len(result.metadata.get('source_nodes', [])))
                            
                            return result
                        except Exception as e:
                            span.set_metadata("error", str(e))
                            raise
                            
                return async_wrapper
            else:
                @functools.wraps(original_method)
                def sync_wrapper(self_obj, *args, **kwargs):
                    return original_method(self_obj, *args, **kwargs)
                return sync_wrapper

        patterns = [
            ("llama_index.query_engine.base", "BaseQueryEngine.query"),
            ("llama_index.query_engine.base", "BaseQueryEngine.aquery"),
            ("llama_index.indices.base", "BaseIndex.query"),
            ("llama_index.indices.base", "BaseIndex.aquery"),
        ]
        
        for module_name, method_path in patterns:
            self._patch_method(module_name, method_path, create_llamaindex_wrapper)

    def instrument_requests(self):
        """Instrument requests library for HTTP tool executions"""
        
        def create_requests_wrapper(original_method):
            @functools.wraps(original_method)
            def sync_wrapper(*args, **kwargs):
                # Extract URL and method
                url = args[0] if args else kwargs.get('url', 'unknown')
                method = original_method.__name__.upper()
                
                # Create event using sync pattern (simplified)
                start_time = asyncio.get_event_loop().time() if asyncio.get_event_loop().is_running() else None
                
                try:
                    result = original_method(*args, **kwargs)
                    
                    # Would need to implement sync event creation or queue for async sending
                    # For now, just return result
                    return result
                    
                except Exception as e:
                    raise
                    
            return sync_wrapper

        patterns = [
            ("requests", "get"),
            ("requests", "post"),
            ("requests", "put"),
            ("requests", "delete"),
            ("requests", "patch"),
        ]
        
        for module_name, method_path in patterns:
            self._patch_method(module_name, method_path, create_requests_wrapper)

    def _estimate_openai_cost(self, model: str, total_tokens: int) -> Optional[float]:
        """Estimate cost for OpenAI models (rough estimates)"""
        # These are example rates - you'd want to keep these updated
        rates_per_1k = {
            'gpt-4': 0.03,
            'gpt-4-32k': 0.06,
            'gpt-3.5-turbo': 0.002,
            'gpt-3.5-turbo-16k': 0.004,
            'text-davinci-003': 0.02,
            'text-curie-001': 0.002,
            'text-babbage-001': 0.0005,
            'text-ada-001': 0.0004,
        }
        
        for model_prefix, rate in rates_per_1k.items():
            if model.startswith(model_prefix):
                return (total_tokens / 1000) * rate
        
        return None

