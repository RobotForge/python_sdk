"""
Custom exceptions for the Telemetry SDK
"""


class TelemetrySDKError(Exception):
    """Base exception for all Telemetry SDK errors"""
    pass


class ConfigurationError(TelemetrySDKError):
    """Raised when there's a configuration issue"""
    pass


class AuthenticationError(TelemetrySDKError):
    """Raised when authentication fails"""
    pass


class NetworkError(TelemetrySDKError):
    """Raised when network operations fail"""
    pass


class ValidationError(TelemetrySDKError):
    """Raised when data validation fails"""
    pass


class BatchError(TelemetrySDKError):
    """Raised when batch operations fail"""
    pass


class InstrumentationError(TelemetrySDKError):
    """Raised when auto-instrumentation fails"""
    pass


class TimeoutError(TelemetrySDKError):
    """Raised when operations timeout"""
    pass


class PayloadTooLargeError(TelemetrySDKError):
    """Raised when payload exceeds size limits"""
    pass


class RateLimitError(TelemetrySDKError):
    """Raised when rate limits are exceeded"""
    pass