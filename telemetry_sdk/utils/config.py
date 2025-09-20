"""
Configuration utilities for the Telemetry SDK
"""

import os
import json
from typing import Dict, Any, Optional
from pathlib import Path

from .exceptions import ConfigurationError
from ..client.models import TelemetryConfig


class ConfigManager:
    """Manages configuration for the Telemetry SDK"""
    
    # Environment variable mappings
    ENV_MAPPINGS = {
        'api_key': 'TELEMETRY_API_KEY',
        'endpoint': 'TELEMETRY_ENDPOINT',
        'project_id': 'TELEMETRY_PROJECT_ID',
        'tenant_id': 'TELEMETRY_TENANT_ID',
        'user_id': 'TELEMETRY_USER_ID',
        'application_id': 'TELEMETRY_APPLICATION_ID',
        'session_id': 'TELEMETRY_SESSION_ID',
        'auto_send': 'TELEMETRY_AUTO_SEND',
        'batch_size': 'TELEMETRY_BATCH_SIZE',
        'batch_timeout': 'TELEMETRY_BATCH_TIMEOUT',
        'pii_scrubbing': 'TELEMETRY_PII_SCRUBBING',
        'max_payload_size': 'TELEMETRY_MAX_PAYLOAD_SIZE',
        'retry_attempts': 'TELEMETRY_RETRY_ATTEMPTS',
        'request_timeout': 'TELEMETRY_REQUEST_TIMEOUT',
    }
    
    # Default configuration
    DEFAULT_CONFIG = {
        'tenant_id': 'default',
        'user_id': 'default',
        'application_id': 'default',
        'auto_send': True,
        'batch_size': 50,
        'batch_timeout': 5.0,
        'pii_scrubbing': False,
        'max_payload_size': 100_000,
        'retry_attempts': 3,
        'request_timeout': 30.0,
    }

    @classmethod
    def load_config(
        cls,
        config_file: Optional[str] = None,
        **override_params
    ) -> TelemetryConfig:
        """
        Load configuration from multiple sources with precedence:
        1. override_params (highest priority)
        2. Environment variables
        3. Configuration file
        4. Defaults (lowest priority)
        """
        config_data = cls.DEFAULT_CONFIG.copy()
        
        # Load from file if provided
        if config_file:
            file_config = cls._load_from_file(config_file)
            config_data.update(file_config)
        
        # Load from environment variables
        env_config = cls._load_from_env()
        config_data.update(env_config)
        
        # Apply overrides
        config_data.update(override_params)
        
        # Validate required fields
        cls._validate_config(config_data)
        
        return TelemetryConfig(**config_data)

    @classmethod
    def _load_from_file(cls, config_file: str) -> Dict[str, Any]:
        """Load configuration from JSON or YAML file"""
        config_path = Path(config_file)
        
        if not config_path.exists():
            raise ConfigurationError(f"Configuration file not found: {config_file}")
        
        try:
            with open(config_path, 'r') as f:
                if config_path.suffix.lower() in ['.yml', '.yaml']:
                    try:
                        import yaml
                        return yaml.safe_load(f)
                    except ImportError:
                        raise ConfigurationError(
                            "PyYAML is required to load YAML configuration files. "
                            "Install with: pip install PyYAML"
                        )
                elif config_path.suffix.lower() == '.json':
                    return json.load(f)
                else:
                    raise ConfigurationError(
                        f"Unsupported configuration file format: {config_path.suffix}"
                    )
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration file: {e}")

    @classmethod
    def _load_from_env(cls) -> Dict[str, Any]:
        """Load configuration from environment variables"""
        config = {}
        
        for config_key, env_var in cls.ENV_MAPPINGS.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                # Type conversion based on default values
                if config_key in ['auto_send', 'pii_scrubbing']:
                    config[config_key] = env_value.lower() in ('true', '1', 'yes', 'on')
                elif config_key in ['batch_size', 'max_payload_size', 'retry_attempts']:
                    try:
                        config[config_key] = int(env_value)
                    except ValueError:
                        raise ConfigurationError(
                            f"Invalid integer value for {env_var}: {env_value}"
                        )
                elif config_key in ['batch_timeout', 'request_timeout']:
                    try:
                        config[config_key] = float(env_value)
                    except ValueError:
                        raise ConfigurationError(
                            f"Invalid float value for {env_var}: {env_value}"
                        )
                else:
                    config[config_key] = env_value
        
        return config

    @classmethod
    def _validate_config(cls, config: Dict[str, Any]) -> None:
        """Validate configuration parameters"""
        required_fields = ['api_key', 'endpoint', 'project_id']
        
        for field in required_fields:
            if not config.get(field):
                env_var = cls.ENV_MAPPINGS.get(field, field.upper())
                raise ConfigurationError(
                    f"Required configuration field '{field}' is missing. "
                    f"Set it via parameter, environment variable {env_var}, or config file."
                )
        
        # Validate numeric ranges
        if config.get('batch_size', 0) <= 0:
            raise ConfigurationError("batch_size must be greater than 0")
        
        if config.get('batch_timeout', 0) <= 0:
            raise ConfigurationError("batch_timeout must be greater than 0")
        
        if config.get('max_payload_size', 0) <= 0:
            raise ConfigurationError("max_payload_size must be greater than 0")
        
        if config.get('retry_attempts', 0) < 0:
            raise ConfigurationError("retry_attempts must be >= 0")

    @classmethod
    def create_sample_config_file(cls, file_path: str, format: str = 'json') -> None:
        """Create a sample configuration file"""
        sample_config = {
            "api_key": "your-api-key-here",
            "endpoint": "https://your-telemetry-server.com",
            "project_id": "your-project-id",
            "tenant_id": "your-tenant-id",
            "user_id": "your-user-id",
            "application_id": "your-app-id",
            "auto_send": True,
            "batch_size": 50,
            "batch_timeout": 5.0,
            "pii_scrubbing": False,
            "max_payload_size": 100000,
            "retry_attempts": 3,
            "request_timeout": 30.0
        }
        
        config_path = Path(file_path)
        
        try:
            with open(config_path, 'w') as f:
                if format.lower() in ['yml', 'yaml']:
                    try:
                        import yaml
                        yaml.dump(sample_config, f, default_flow_style=False)
                    except ImportError:
                        raise ConfigurationError(
                            "PyYAML is required to create YAML configuration files. "
                            "Install with: pip install PyYAML"
                        )
                elif format.lower() == 'json':
                    json.dump(sample_config, f, indent=2)
                else:
                    raise ConfigurationError(f"Unsupported format: {format}")
            
            print(f"Sample configuration file created: {config_path}")
            
        except Exception as e:
            raise ConfigurationError(f"Failed to create configuration file: {e}")


def load_config(**kwargs) -> TelemetryConfig:
    """Convenience function to load configuration"""
    return ConfigManager.load_config(**kwargs)


def create_config_file(file_path: str, format: str = 'json') -> None:
    """Convenience function to create sample config file"""
    ConfigManager.create_sample_config_file(file_path, format)