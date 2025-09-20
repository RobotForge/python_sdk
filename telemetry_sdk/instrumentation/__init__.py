"""
Auto-instrumentation module for the Telemetry SDK
"""

from .auto_instrumentation import AutoInstrumentation
from .framework_integrations import FrameworkIntegrations

__all__ = [
    "AutoInstrumentation",
    "FrameworkIntegrations",
]