"""
Enhanced logger with telemetry-specific methods
"""

import logging
from typing import TYPE_CHECKING, Optional

from .handler import TelemetryHandler

if TYPE_CHECKING:
    from ..client.telemetry_client import TelemetryClient


class TelemetryLogger:
    """Enhanced logger class with telemetry-specific methods"""
    
    def __init__(self, name: str, client: 'TelemetryClient'):
        self.logger = logging.getLogger(name)
        self.client = client
        
        # Add telemetry handler if not already present
        if not any(isinstance(h, TelemetryHandler) for h in self.logger.handlers):
            handler = TelemetryHandler(client)
            self.logger.addHandler(handler)

    def model_call(
        self,
        message: str = "Model call executed",
        provider: str = None,
        model: str = None,
        input_text: str = None,
        output_text: str = None,
        token_count: int = None,
        latency_ms: int = None,
        cost: float = None,
        temperature: float = None,
        finish_reason: str = None,
        **kwargs
    ):
        """Log a model call event"""
        extra = {
            'event_type': 'model_call',
            'provider': provider,
            'model_name': model,
            'input_text': input_text,
            'output_text': output_text,
            'token_count': token_count,
            'latency_ms': latency_ms,
            'cost': cost,
            'temperature': temperature,
            'finish_reason': finish_reason,
            **kwargs
        }
        # Remove None values
        extra = {k: v for k, v in extra.items() if v is not None}
        self.logger.info(message, extra=extra)

    def tool_execution(
        self,
        message: str = "Tool executed",
        tool_name: str = None,
        action: str = None,
        endpoint: str = None,
        http_method: str = None,
        http_status_code: int = None,
        latency_ms: int = None,
        request_payload: dict = None,
        response_payload: dict = None,
        external_latency_ms: int = None,
        **kwargs
    ):
        """Log a tool execution event"""
        extra = {
            'event_type': 'tool_execution',
            'tool_name': tool_name,
            'action': action,
            'endpoint': endpoint,
            'http_method': http_method,
            'http_status_code': http_status_code,
            'latency_ms': latency_ms,
            'request_payload': request_payload,
            'response_payload': response_payload,
            'external_latency_ms': external_latency_ms,
            **kwargs
        }
        extra = {k: v for k, v in extra.items() if v is not None}
        self.logger.info(message, extra=extra)

    def agent_action(
        self,
        message: str = "Agent action executed",
        action_type: str = None,
        agent_name: str = None,
        thought_process: str = None,
        selected_tool: str = None,
        target_model: str = None,
        policy: dict = None,
        **kwargs
    ):
        """Log an agent action event"""
        extra = {
            'event_type': 'agent_action',
            'action_type': action_type,
            'agent_name': agent_name,
            'thought_process': thought_process,
            'selected_tool': selected_tool,
            'target_model': target_model,
            'policy': policy,
            **kwargs
        }
        extra = {k: v for k, v in extra.items() if v is not None}
        self.logger.info(message, extra=extra)

    def mcp_event(
        self,
        message: str = "MCP event executed", 
        mcp_version: str = None,
        endpoint: str = None,
        request_type: str = None,
        request_payload: dict = None,
        response_payload: dict = None,
        roundtrip_latency_ms: int = None,
        **kwargs
    ):
        """Log an MCP event"""
        extra = {
            'event_type': 'mcp_event',
            'mcp_version': mcp_version,
            'endpoint': endpoint,
            'request_type': request_type,
            'request_payload': request_payload,
            'response_payload': response_payload,
            'roundtrip_latency_ms': roundtrip_latency_ms,
            **kwargs
        }
        extra = {k: v for k, v in extra.items() if v is not None}
        self.logger.info(message, extra=extra)

    def error(self, message: str, **kwargs):
        """Log an error"""
        self.logger.error(message, extra=kwargs)

    def warning(self, message: str, **kwargs):
        """Log a warning"""
        self.logger.warning(message, extra=kwargs)

    def info(self, message: str, **kwargs):
        """Log info message"""
        self.logger.info(message, extra=kwargs)

    def debug(self, message: str, **kwargs):
        """Log debug message"""
        self.logger.debug(message, extra=kwargs)

    def set_level(self, level: int):
        """Set logging level"""
        self.logger.setLevel(level)

    def add_handler(self, handler: logging.Handler):
        """Add logging handler"""
        self.logger.addHandler(handler)

    def remove_handler(self, handler: logging.Handler):
        """Remove logging handler"""
        self.logger.removeHandler(handler)


def configure_telemetry_logging(
    client: 'TelemetryClient',
    logger_name: str = None,
    level: int = logging.INFO,
    format_string: str = None,
    add_console_handler: bool = True
) -> TelemetryLogger:
    """Configure telemetry logging for an application"""
    
    logger_name = logger_name or "telemetry"
    
    # Create telemetry logger
    telemetry_logger = TelemetryLogger(logger_name, client)
    
    # Set logging level
    telemetry_logger.set_level(level)
    
    # Add console handler if requested and no handlers exist
    if add_console_handler and not telemetry_logger.logger.handlers:
        console_handler = logging.StreamHandler()
        
        if format_string:
            formatter = logging.Formatter(format_string)
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        
        console_handler.setFormatter(formatter)
        telemetry_logger.add_handler(console_handler)
    
    return telemetry_logger


def setup_telemetry_logging(
    api_key: str,
    endpoint: str,
    project_id: str,
    logger_name: str = "telemetry",
    **client_kwargs
) -> TelemetryLogger:
    """Quick setup for telemetry logging"""
    
    from ..client.telemetry_client import TelemetryClient
    
    client = TelemetryClient(
        api_key=api_key,
        endpoint=endpoint,
        project_id=project_id,
        **client_kwargs
    )
    
    return configure_telemetry_logging(client, logger_name)