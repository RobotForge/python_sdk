# src/telemetry/robotforge_exporter.py (FINAL FIX)

"""
RobotForge SpanExporter - Custom OpenTelemetry exporter using RobotForge SDK.
"""

import logging
from typing import Optional, Sequence, Dict, Any
from datetime import datetime, timezone

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.trace import StatusCode
import asyncio
import threading

# Import from RobotForge SDK
try:
    from telemetry_sdk.client.telemetry_client import TelemetryClient
    from telemetry_sdk.client.event_builder import (
        EventBuilder,
        ModelCallEventBuilder,
        ToolExecutionEventBuilder,
        AgentActionEventBuilder
    )
    from telemetry_sdk.client.models import EventType, EventStatus
except ImportError:
    raise ImportError(
        "RobotForge SDK not installed. Install with: pip install robotforge-python-sdk"
    )

logger = logging.getLogger(__name__)


class SpanExporter(SpanExporter):
    """Custom OpenTelemetry SpanExporter that uses RobotForge SDK client."""
    
    def __init__(
        self,
        robotforge_client: TelemetryClient,
        enable_console_logging: bool = False,
        map_resource_attributes: bool = True
    ):
        self.client = robotforge_client
        self.enable_console_logging = enable_console_logging
        self.map_resource_attributes = map_resource_attributes
        
        print(
            f"RobotForgeSpanExporter initialized "
            f"(app={robotforge_client.config.application_id})"
        )

    def run_coro_sync_safe(self, coro):
        if not asyncio.iscoroutine(coro):
            return coro

        try:
            loop = asyncio.get_running_loop()
            # Running loop exists: schedule background task
            asyncio.create_task(coro)
        except RuntimeError:
            # No running loop: run in separate thread to avoid closing main loop
            def thread_runner():
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    new_loop.run_until_complete(coro)
                finally:
                    # Do NOT set loop to None here in a thread, just close
                    new_loop.close()

            t = threading.Thread(target=thread_runner, daemon=True)
            t.start()
            t.join()

    # ---------------- Export ---------------- #
    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        """Export spans to RobotForge using the SDK client."""
        if not spans:
            return SpanExportResult.SUCCESS

        try:
            for span in spans:
                try:
                    # Convert OTEL span to RobotForge EventBuilder
                    event_builder = self._span_to_event_builder(span)
                    event = event_builder.build()
                    details = {}

                    # Queue event async-safe
                    self._send_event_sync_safe(event, details)

                    if self.enable_console_logging:
                        trace_id_short = format(span.get_span_context().trace_id, '032x')[:8]
                        print(f"✓ Queued: {span.name} [trace={trace_id_short}...]")

                except Exception as e:
                    print(f"Failed to convert span {span.name}: {e}", flush=True)

            print(f"Queued {len(spans)} spans", flush=True)
            return SpanExportResult.SUCCESS

        except Exception as e:
            print(f"Failed to export spans: {e}", flush=True)
            return SpanExportResult.FAILURE


    # ---------------- _send_event_sync_safe ---------------- #
    def _send_event_sync_safe(self, event, details):
        """Helper to send event from sync code safely, handling async send_event."""
        send_coro = self.client.send_event(event, details, immediate=True)
        self.run_coro_sync_safe(send_coro)


    # ---------------- shutdown ---------------- #
    def shutdown(self) -> None:
        """Shutdown exporter and flush pending events safely."""
        try:
            flush_coro = getattr(self.client, "flush", None)
            if flush_coro:
                self.run_coro_sync_safe(flush_coro())
            print("RobotForgeSpanExporter shutdown completed", flush=True)
        except Exception as e:
            print(f"Error during shutdown: {e}", flush=True)


    # ---------------- force_flush ---------------- #
    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Force flush any buffered spans, async-safe."""
        try:
            flush_coro = getattr(self.client, "flush", None)
            if flush_coro:
                self.run_coro_sync_safe(flush_coro())
            return True
        except Exception as e:
            print(f"Error during force flush: {e}", flush=True)
            return False

    def _span_to_event_builder(self, span: ReadableSpan) -> EventBuilder:
        """Convert OpenTelemetry ReadableSpan to RobotForge EventBuilder."""
        
        # Extract span context
        span_context = span.get_span_context()
        trace_id = format(span_context.trace_id, '032x')
        span_id = format(span_context.span_id, '016x')
        parent_span_id = format(span.parent.span_id, '016x') if span.parent else None
        
        # Extract attributes
        attributes = dict(span.attributes) if span.attributes else {}
        
        # Determine event type
        event_type = self._infer_event_type(span.name, attributes)
        
        # Extract service name
        service_name = 'unknown'
        if span.resource and span.resource.attributes:
            service_name = span.resource.attributes.get('service.name', 'unknown')
        
        # Create appropriate EventBuilder
        if event_type == EventType.MODEL_CALL:
            builder = ModelCallEventBuilder(
                client=self.client,
                source_component=service_name
            )
        elif event_type == EventType.TOOL_EXECUTION:
            tool_name = attributes.get('tool_name', span.name)
            builder = ToolExecutionEventBuilder(
                client=self.client,
                tool_name=tool_name,
                source_component=service_name
            )
        elif event_type == EventType.AGENT_ACTION:
            action_type = attributes.get('action_type', span.name)
            builder = AgentActionEventBuilder(
                client=self.client,
                action_type=action_type,
                source_component=service_name
            )
        else:
            builder = EventBuilder(
                client=self.client,
                event_type=event_type,
                source_component=service_name
            )
        
        # ✅ FIX: Use set_trace_info instead of set_trace_context
        builder.set_trace_info(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id
        )
        
        # Set operation_name and service_name on the event directly
        if hasattr(builder, '_event'):
            builder._event.operation_name = span.name
            builder._event.service_name = service_name
        
        # Extract and set input/output
        input_text = self._extract_input(attributes)
        output_text = self._extract_output(attributes)
        
        if input_text:
            builder.set_input(input_text)
        if output_text:
            builder.set_output(output_text)
        
        # Calculate and set latency
        if span.start_time and span.end_time:
            duration_ns = span.end_time - span.start_time
            latency_ms = int(duration_ns / 1_000_000)
            builder.set_latency(latency_ms)
        
        # Set status
        if span.status.status_code == StatusCode.ERROR:
            builder.set_status(EventStatus.ERROR)
            if span.status.description:
                builder.set_metadata('error_description', span.status.description)
        else:
            builder.set_status(EventStatus.SUCCESS)
        
        # Extract and set token count
        token_count = self._extract_token_count(attributes)
        if token_count:
            builder.set_tokens(token_count)
        
        # Extract and set cost
        cost = self._extract_cost(attributes)
        if cost:
            builder.set_cost(cost)
        
        # Extract provider and model for MODEL_CALL events
        if event_type == EventType.MODEL_CALL:
            provider = attributes.get('gen_ai.system') or attributes.get('llm.provider')
            model = attributes.get('gen_ai.request.model') or attributes.get('llm.model')
            
            if provider:
                builder.set_provider(provider)
            if model:
                builder.set_model(model)
        

            # Add metadata from attributes
        metadata = self._build_metadata(span, attributes)
        for key, value in metadata.items():
            builder.set_metadata(key, value)
        

        tags = self._extract_tags(attributes)
        if tags and hasattr(builder, '_event'):
            builder._event.tags = tags
        
        # Set timestamp
        if span.start_time:
            timestamp = datetime.fromtimestamp(span.start_time / 1e9, tz=timezone.utc)
            builder.set_timestamp(timestamp)
        
        return builder

    
    def _infer_event_type(self, span_name: str, attributes: Dict[str, Any]) -> EventType:
        """Infer RobotForge event type from span name and attributes."""
        
        # Check for GenAI semantic conventions
        if any(key.startswith('gen_ai.') for key in attributes.keys()):
            return EventType.MODEL_CALL
        
        # Check for HTTP/RPC
        if 'http.method' in attributes or 'rpc.method' in attributes:
            return EventType.TOOL_EXECUTION
        
        # Check span name
        span_name_lower = span_name.lower()
        if any(kw in span_name_lower for kw in ['llm', 'model', 'chat', 'completion', 'generation']):
            return EventType.MODEL_CALL
        elif any(kw in span_name_lower for kw in ['tool', 'function', 'http', 'api', 'execute']):
            return EventType.TOOL_EXECUTION
        elif any(kw in span_name_lower for kw in ['agent', 'action', 'decision']):
            return EventType.AGENT_ACTION
        
        return EventType.MODEL_CALL
    
    def _extract_input(self, attributes: Dict[str, Any]) -> Optional[str]:
        """Extract input text from attributes."""
        for key in ['gen_ai.prompt', 'gen_ai.request.prompt', 'input', 'prompt', 'query']:
            if key in attributes:
                return str(attributes[key])
        return None
    
    def _extract_output(self, attributes: Dict[str, Any]) -> Optional[str]:
        """Extract output text from attributes."""
        for key in ['gen_ai.completion', 'gen_ai.response.completion', 'output', 'response']:
            if key in attributes:
                return str(attributes[key])
        return None
    
    def _extract_token_count(self, attributes: Dict[str, Any]) -> Optional[int]:
        """Extract token count from attributes."""
        if 'gen_ai.usage.total_tokens' in attributes:
            return int(attributes['gen_ai.usage.total_tokens'])
        
        input_tokens = attributes.get('gen_ai.usage.input_tokens', 0)
        output_tokens = attributes.get('gen_ai.usage.output_tokens', 0)
        
        if input_tokens or output_tokens:
            return int(input_tokens) + int(output_tokens)
        
        return None
    
    def _extract_cost(self, attributes: Dict[str, Any]) -> Optional[float]:
        """Extract cost from attributes."""
        if 'gen_ai.usage.cost' in attributes:
            return float(attributes['gen_ai.usage.cost'])
        return None
    
    def _build_metadata(self, span: ReadableSpan, attributes: Dict[str, Any]) -> Dict[str, Any]:
        """Build metadata dictionary from span and attributes."""
        metadata = {'otel_source': True, 'span_kind': str(span.kind)}
        
        if self.map_resource_attributes and span.resource and span.resource.attributes:
            metadata['resource'] = {k: str(v) for k, v in span.resource.attributes.items()}
        
        if span.instrumentation_scope:
            metadata['instrumentation_scope'] = {
                'name': span.instrumentation_scope.name,
                'version': span.instrumentation_scope.version or '',
            }
        
        # GenAI attributes
        gen_ai_attrs = {k: v for k, v in attributes.items() if k.startswith('gen_ai.')}
        if gen_ai_attrs:
            metadata['gen_ai'] = gen_ai_attrs
        
        # Usage details
        if 'gen_ai.usage.input_tokens' in attributes:
            metadata['usage_details'] = {
                'prompt_tokens': int(attributes.get('gen_ai.usage.input_tokens', 0)),
                'completion_tokens': int(attributes.get('gen_ai.usage.output_tokens', 0)),
                'total_tokens': self._extract_token_count(attributes) or 0
            }
        
        # HTTP attributes
        http_attrs = {k: v for k, v in attributes.items() if k.startswith('http.')}
        if http_attrs:
            metadata['http'] = http_attrs
        
        # Events (exceptions)
        if span.events:
            metadata['events'] = [
                {
                    'name': event.name,
                    'timestamp': event.timestamp,
                    'attributes': dict(event.attributes) if event.attributes else {}
                }
                for event in span.events
            ]
        
        return metadata
    
    def _extract_tags(self, attributes: Dict[str, Any]) -> Dict[str, str]:
        """Extract tags from attributes."""
        tags = {}
        tag_keys = [
            'environment', 'deployment.environment', 'service.version',
            'gen_ai.system', 'gen_ai.request.model', 'http.method', 'http.status_code'
        ]
        
        for key in tag_keys:
            if key in attributes:
                tags[key] = str(attributes[key])
        
        return tags
    



def create_span_exporter(
    api_key: str,
    user_id: str,
    application_id: str = "otel-app",
    endpoint: str = "https://cloud.robotforge.com.ng",
    **client_kwargs
) -> SpanExporter:
    """Create a RobotForgeSpanExporter with a new TelemetryClient."""
    from telemetry_sdk import TelemetryClient
    
    client = TelemetryClient(
        api_key=api_key,
        user_id=user_id,
        application_id=application_id,
        endpoint=endpoint,
        **client_kwargs
    )
    
    return SpanExporter(client, enable_console_logging=True)