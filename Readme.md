# Telemetry SDK

A comprehensive, multi-layered Python SDK for AI/ML telemetry that supports context managers, decorators, auto-instrumentation, and logging integration. Monitor your AI applications with zero or minimal code changes.

## 🚀 Features

- **Multiple Integration Patterns**: Choose from context managers, decorators, auto-instrumentation, or logging
- **Zero-Code Auto-Instrumentation**: Automatically trace OpenAI, Anthropic, LangChain, and other popular libraries
- **Async-First Design**: Built for modern async Python applications
- **Rich Event Types**: Support for model calls, tool executions, agent actions, and MCP events
- **Framework Integrations**: Built-in support for FastAPI, Flask, Django
- **OpenTelemetry Compatible**: Works with existing OpenTelemetry infrastructure
- **Production Ready**: Robust error handling, batching, and retry logic

## 📦 Installation

```bash
# Basic installation
pip install telemetry-sdk

# With auto-instrumentation support
pip install telemetry-sdk[auto]

# With all optional dependencies
pip install telemetry-sdk[all]
```

## 🏃 Quick Start

### 1. Basic Setup

```python
import telemetry_sdk

# Quick setup with sensible defaults
client = telemetry_sdk.quick_setup(
    api_key="your-api-key",
    endpoint="https://your-telemetry-server.com",
    project_id="my-ai-project"
)
```

### 2. Context Manager Approach (Recommended)

```python
# Model call tracing
async with client.trace_model_call(provider="openai", model="gpt-4") as span:
    span.set_input("What is AI?")
    response = await openai_client.chat.completions.create(...)
    span.set_output(response.choices[0].message.content)
    span.set_tokens(response.usage.total_tokens)

# Tool execution tracing  
async with client.trace_tool_execution(tool_name="web_search") as span:
    results = await search_api.search("AI news")
    span.set_output(str(results))

# Agent action tracing
async with client.trace_agent_action(action_type="planning") as span:
    plan = await create_execution_plan()
    span.set_metadata("steps", len(plan.steps))
```

### 3. Decorator Approach

```python
@client.trace_model_call_decorator(provider="openai", model="gpt-4")
async def generate_response(prompt: str):
    response = await openai_client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

@client.trace_tool_execution_decorator(tool_name="calculator")
async def calculate(expression: str):
    return eval(expression)  # Don't do this in production!
```

### 4. Auto-Instrumentation (Zero Code Changes)

```python
from telemetry_sdk import AutoInstrumentation

# Enable auto-instrumentation for popular libraries
auto_instr = AutoInstrumentation(client)
auto_instr.instrument_all()

# Now all calls to instrumented libraries are automatically traced!
response = await openai.ChatCompletion.acreate(...)  # Automatically traced
chain_result = langchain_chain.run(...)              # Automatically traced
```

### 5. Logging Integration

```python
import telemetry_sdk

# Setup telemetry logging
logger = telemetry_sdk.setup_telemetry_logging(
    api_key="your-api-key",
    endpoint="https://your-telemetry-server.com", 
    project_id="my-project"
)

# Use like regular logging, but with telemetry superpowers
logger.model_call(
    "GPT-4 response generated",
    provider="openai",
    model="gpt-4",
    input_text="Hello",
    output_text="Hi there!",
    token_count=10,
    latency_ms=150,
    cost=0.001
)
```

## 🔧 Configuration

### Environment Variables

```bash
export TELEMETRY_API_KEY=your-api-key
export TELEMETRY_ENDPOINT=https://your-server.com
export TELEMETRY_PROJECT_ID=my-project
export TELEMETRY_AUTO_SEND=true
export TELEMETRY_BATCH_SIZE=50
```

### Configuration File

Create `telemetry.yaml`:

```yaml
api_key: your-api-key
endpoint: https://your-server.com
project_id: my-project
auto_send: true
batch_size: 50
batch_timeout: 5.0
retry_attempts: 3
```

Load with:

```python
from telemetry_sdk import load_config, TelemetryClient

config = load_config(config_file="telemetry.yaml")
client = TelemetryClient(config=config)
```

### Programmatic Configuration

```python
from telemetry_sdk import TelemetryClient

client = TelemetryClient(
    api_key="your-key",
    endpoint="https://your-server.com",
    project_id="my-project",
    auto_send=True,           # Enable background batching
    batch_size=50,            # Events per batch
    batch_timeout=5.0,        # Max time between sends
    retry_attempts=3,         # Retry failed requests
    max_payload_size=100_000  # Max event size
)
```

## 🔗 OpenTelemetry Integration

The SDK is fully compatible with OpenTelemetry and can work alongside your existing observability infrastructure.

### Option 1: Use with OpenTelemetry Collector

```python
import telemetry_sdk
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Setup OpenTelemetry
trace.set_tracer_provider(TracerProvider())
otlp_exporter = OTLPSpanExporter(endpoint="http://otel-collector:4317")
span_processor = BatchSpanProcessor(otlp_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)

# Setup Telemetry SDK (can coexist)
client = telemetry_sdk.quick_setup(
    api_key="your-key",
    endpoint="https://your-telemetry-server.com",
    project_id="my-project"
)

# Both systems will capture telemetry
async with client.trace_model_call() as span:
    # This creates both telemetry-sdk events AND OpenTelemetry spans
    response = await openai_call()
```

### Option 2: Export to OpenTelemetry

```python
# Configure telemetry server to export to OpenTelemetry
# Set this in your telemetry server environment
export OTLP_ENDPOINT=http://your-otel-collector:4317

# Your telemetry server will forward traces to OpenTelemetry
client = telemetry_sdk.quick_setup(
    api_key="your-key", 
    endpoint="https://your-telemetry-server.com",
    project_id="my-project"
)
```

### Option 3: Trace Correlation

```python
from opentelemetry import trace

# Get current OpenTelemetry span context
current_span = trace.get_current_span()
trace_id = format(current_span.get_span_context().trace_id, '032x')
span_id = format(current_span.get_span_context().span_id, '016x')

# Include in telemetry event
async with client.trace_model_call() as span:
    span.set_metadata("otel_trace_id", trace_id)
    span.set_metadata("otel_span_id", span_id)
    # Your operation here
```

## 📊 Event Types

The SDK supports four main event types:

- **Model Calls**: LLM API calls (OpenAI, Anthropic, etc.)
- **Tool Executions**: External API calls, database queries, etc.
- **Agent Actions**: Planning, reasoning, decision-making
- **MCP Events**: Model Context Protocol events

## 🔌 Framework Integrations

### FastAPI

```python
from fastapi import FastAPI
from telemetry_sdk import FrameworkIntegrations

app = FastAPI()
integrations = FrameworkIntegrations(client)
integrations.wrap_fastapi_app(app)  # All endpoints automatically traced
```

### LangChain

```python
from telemetry_sdk import FrameworkIntegrations

# Wrap existing chains
integrations = FrameworkIntegrations(client)
traced_chain = integrations.wrap_langchain_chain(your_chain)
```

### LlamaIndex

```python
# Wrap query engines
traced_engine = integrations.wrap_llamaindex_query_engine(query_engine)
result = traced_engine.query("What is AI?")  # Automatically traced
```

## 🎯 Auto-Instrumentation Support

The SDK automatically instruments these libraries when available:

- **OpenAI**: `openai` package
- **Anthropic**: `anthropic` package  
- **LangChain**: `langchain` package
- **LlamaIndex**: `llama-index` package
- **HTTP Libraries**: `requests`, `httpx`

Enable with:

```python
from telemetry_sdk import AutoInstrumentation

auto_instr = AutoInstrumentation(client)
auto_instr.instrument_all()

# Check what was instrumented
status = auto_instr.get_instrumentation_status()
print(status)  # {"openai": True, "langchain": True, ...}
```

## 🧪 Examples

Check out the `examples/` directory for comprehensive examples:

- `basic_usage.py` - Core SDK patterns
- `advanced_agent.py` - Complex AI agent with nested tracing
- `fastapi_app.py` - Web API with telemetry integration
- `auto_instrumentation_example.py` - Zero-code instrumentation
- `logging_example.py` - Logging-based telemetry

## ⚡ Performance

- **Async-first**: Non-blocking telemetry operations
- **Auto-batching**: Efficient bulk sending with configurable limits
- **Circuit breaker**: Automatic fallback if telemetry service is down
- **Memory efficient**: Minimal overhead on your application

## 🛡️ Error Handling

The SDK is designed to never break your application:

```python
# Telemetry failures are isolated
async with client.trace_model_call() as span:
    # Even if telemetry fails, your code continues
    result = await your_critical_operation()
    return result  # This will always work
```

## 🔧 Development

### Setup Development Environment

```bash
git clone https://github.com/yourorg/telemetry-sdk
cd telemetry-sdk
pip install -e ".[dev,auto]"
pre-commit install
```

### Run Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=telemetry_sdk --cov-report=html

# Run specific test types
pytest -m unit
pytest -m integration
```

### Build Package

```bash
python -m build
twine check dist/*
```

## 📚 Documentation

- [API Reference](https://telemetry-sdk.readthedocs.io/en/latest/api.html)
- [Integration Guides](https://telemetry-sdk.readthedocs.io/en/latest/integrations.html)
- [Configuration](https://telemetry-sdk.readthedocs.io/en/latest/configuration.html)
- [Examples](https://github.com/yourorg/telemetry-sdk/tree/main/examples)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- [GitHub Issues](https://github.com/yourorg/telemetry-sdk/issues)
- [Documentation](https://telemetry-sdk.readthedocs.io/)
- [Discussions](https://github.com/yourorg/telemetry-sdk/discussions)

## 🗺️ Roadmap

- [ ] Additional auto-instrumentation targets (Hugging Face, etc.)
- [ ] GraphQL and gRPC support
- [ ] Real-time streaming capabilities
- [ ] Enhanced PII detection and scrubbing
- [ ] Custom metric aggregations
- [ ] Distributed tracing enhancements

---

Made with ❤️ for the AI/ML community