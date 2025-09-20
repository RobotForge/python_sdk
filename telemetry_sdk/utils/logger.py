
import logging
from client.telemetry_client import TelemetryClient
from utils.handler import TelemetryHandler


class TelemetryLogger:
    """Enhanced logger class with telemetry-specific methods"""
    
    def __init__(self, name: str, client: TelemetryClient):
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
        roundtrip_latency_ms: int = None,
        **kwargs
    ):
        """Log an MCP event"""
        extra = {
            'event_type': 'mcp_event',
            'mcp_version': mcp_version,
            'endpoint': endpoint,
            'request_type': request_type,
            'roundtrip_latency_ms': roundtrip_latency_ms,
            **kwargs
        }
        extra = {k: v for k, v in extra.items() if v is not None}
        self.logger.info(message, extra=extra)

    def error(self, message: str, **kwargs):
        """Log an error"""
        extra = kwargs
        self.logger.error(message, extra=extra)

    def info(self, message: str, **kwargs):
        """Log info message"""
        extra = kwargs
        self.logger.info(message, extra=extra)

    def debug(self, message: str, **kwargs):
        """Log debug message"""
        extra = kwargs
        self.logger.debug(message, extra=extra)


def configure_telemetry_logging(
    client: TelemetryClient,
    logger_name: str = None,
    level: int = logging.INFO,
    format_string: str = None
) -> TelemetryLogger:
    """Configure telemetry logging for an application"""
    
    logger_name = logger_name or "telemetry"
    
    # Create telemetry logger
    telemetry_logger = TelemetryLogger(logger_name, client)
    
    # Set logging level
    telemetry_logger.logger.setLevel(level)
    
    # Add console handler if no handlers exist
    if not telemetry_logger.logger.handlers:
        console_handler = logging.StreamHandler()
        if format_string:
            formatter = logging.Formatter(format_string)
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        console_handler.setFormatter(formatter)
        telemetry_logger.logger.addHandler(console_handler)
    
    return telemetry_logger


# Convenience function for quick setup
def setup_telemetry_logging(
    api_key: str,
    endpoint: str,
    project_id: str,
    logger_name: str = "telemetry",
    **client_kwargs
) -> TelemetryLogger:
    """Quick setup for telemetry logging"""
    
    client = TelemetryClient(
        api_key=api_key,
        endpoint=endpoint,
        project_id=project_id,
        **client_kwargs
    )
    
    return configure_telemetry_logging(client, logger_name)