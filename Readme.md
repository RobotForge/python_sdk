# RobotForge Python SDK Documentation

Build trust and reliability into your AI applications with comprehensive telemetry and observability.

---

## 🎯 Introduction

The RobotForge Python SDK is your first step toward building trustworthy and reliable AI applications. In a world where AI systems make critical decisions, having complete visibility into your model calls, tool executions, and agent behaviors isn't just nice to have—it's essential.

### What RobotForge Telemetry Actually Helps You Do

**Debug with Confidence**: When your AI application behaves unexpectedly, pinpoint exactly where things went wrong. See the complete chain of model calls, tool executions, and decision points that led to any outcome.

**Optimize Performance**: Identify bottlenecks in your AI workflows. Track token usage, response times, and costs across different models and configurations to make data-driven optimization decisions.

**Build Trust**: Provide stakeholders with complete transparency into your AI systems. Show exactly what inputs were processed, what decisions were made, and what outputs were generated—critical for compliance, auditing, and accountability.

**Monitor in Production**: Detect issues before your users do. Real-time monitoring of model performance, error rates, and unexpected behaviors helps you maintain high-quality AI experiences.

**Understand Usage Patterns**: Gain insights into how your AI features are actually being used. Which models are most popular? Which tools get called most frequently? Where are users experiencing friction?

The RobotForge SDK makes all of this possible with minimal code changes and zero impact on your application's performance.

---

## 📦 Installation

```bash
pip install robotforge-python-sdk
```

**Requirements:**
- Python 3.8+
- `aiohttp` for async HTTP requests
- `requests` for synchronous operations

---

## 🚀 Quick Start

### Basic Setup

```python
import telemetery_sdk

# Initialize the SDK
client = TelemetryClient(
    api_key="your-api-key"
)
```

### Your First Trace

```python
import asyncio
from telemetry_sdk.client import TelemetryClient

async def main():
    client = TelmetryClient(
        api_key="your-api-key"
    )
    
    # Trace a model call
    async with client.trace_model_call(
        provider="openai",
        model="gpt-4"
    ) as span:
        span.set_input("What is the capital of France?")
        
        # Your AI logic here
        response = "Paris"
        
        span.set_output(response)
        span.set_tokens(150)

asyncio.run(main())
```

---

## 🏗️ Core Features

### 1. Model Call Tracing

Track every interaction with your language models—inputs, outputs, tokens, costs, and performance metrics.

#### Async Context Manager (Recommended)

```python
async with client.trace_model_call(
    provider="openai",
    model="gpt-4-turbo",
    temperature=0.7
) as span:
    span.set_input(user_prompt)
    
    # Make your model call
    response = await openai.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": user_prompt}],
        temperature=0.7
    )
    
    # Capture the results
    span.set_output(response.choices[0].message.content)
    span.set_tokens(response.usage.total_tokens)
    span.set_cost(calculate_cost(response.usage))
```

#### Synchronous Context Manager

```python
with client.trace_model_call_sync(
    provider="anthropic",
    model="claude-3-sonnet"
) as span:
    span.set_input(prompt)
    
    # Synchronous model call
    response = anthropic.messages.create(
        model="claude-3-sonnet-20240229",
        messages=[{"role": "user", "content": prompt}]
    )
    
    span.set_output(response.content[0].text)
    span.set_tokens(response.usage.input_tokens + response.usage.output_tokens)
```

#### Decorator Pattern

```python
@client.trace_model_call_decorator(
    provider="openai",
    model="gpt-4"
)
async def generate_summary(text: str) -> str:
    response = await openai.chat.completions.create(
        model="gpt-4",
        messages=[{
            "role": "user",
            "content": f"Summarize this text: {text}"
        }]
    )
    return response.choices[0].message.content

# Use it like any other function
summary = await generate_summary(long_article)
```

**Available Methods:**
- `set_input(text)` - Set the input prompt or message
- `set_output(text)` - Set the model's response
- `set_tokens(count)` - Set total token count
- `set_cost(amount)` - Set the cost in your currency
- `set_temperature(temp)` - Set the temperature parameter
- `set_provider(name)` - Set the model provider
- `set_model(name)` - Set the model name
- `set_metadata(key, value)` - Add custom metadata

---

### 2. Tool Execution Tracing

Monitor external tool calls, API requests, database queries, and any other operations your AI agents perform.

#### Async Context Manager

```python
async with client.trace_tool_execution(
    tool_name="web_search",
    action="search",
    endpoint="https://api.search.com/v1/search",
    http_method="POST"
) as span:
    span.set_input(search_query)
    
    # Execute your tool
    results = await search_api.search(
        query=search_query,
        limit=10
    )
    
    span.set_output(str(results))
    span.set_http_status_code(200)
```

#### Synchronous Context Manager

```python
with client.trace_tool_execution_sync(
    tool_name="database_query",
    action="fetch_user_data"
) as span:
    span.set_input(f"user_id: {user_id}")
    
    # Execute database query
    user_data = db.query(
        f"SELECT * FROM users WHERE id = {user_id}"
    )
    
    span.set_output(str(user_data))
```

#### Decorator Pattern

```python
@client.trace_tool_execution_decorator(
    tool_name="weather_api",
    action="get_forecast"
)
async def get_weather(city: str) -> dict:
    response = await weather_client.get_forecast(city)
    return response.json()

# Automatically traced
weather = await get_weather("San Francisco")
```

**Available Methods:**
- `set_input(text)` - Set the tool input/parameters
- `set_output(text)` - Set the tool output/results
- `set_action(name)` - Set the action name
- `set_endpoint(url)` - Set the API endpoint
- `set_http_method(method)` - Set HTTP method (GET, POST, etc.)
- `set_http_status_code(code)` - Set HTTP status code
- `set_request_payload(data)` - Set the request payload
- `set_response_payload(data)` - Set the response payload
- `set_metadata(key, value)` - Add custom metadata

---

### 3. Context Managers (Sync & Async)

The SDK provides both synchronous and asynchronous context managers for all tracing operations, giving you flexibility to use them in any Python environment.

#### Async Context Managers

Best for modern async applications using `asyncio`, FastAPI, or other async frameworks.

```python
async def process_request(user_input: str):
    # Model call
    async with client.trace_model_call(
        provider="openai",
        model="gpt-4"
    ) as model_span:
        model_span.set_input(user_input)
        response = await call_model(user_input)
        model_span.set_output(response)
    
    # Tool execution
    async with client.trace_tool_execution(
        tool_name="data_processor"
    ) as tool_span:
        tool_span.set_input(response)
        result = await process_data(response)
        tool_span.set_output(result)
    
    return result
```

#### Synchronous Context Managers

Perfect for traditional synchronous Python applications, scripts, and notebooks.

```python
def analyze_text(text: str):
    # Model call
    with client.trace_model_call_sync(
        provider="anthropic",
        model="claude-3-opus"
    ) as model_span:
        model_span.set_input(text)
        analysis = call_claude(text)
        model_span.set_output(analysis)
    
    # Tool execution
    with client.trace_tool_execution_sync(
        tool_name="sentiment_analyzer"
    ) as tool_span:
        tool_span.set_input(analysis)
        sentiment = analyze_sentiment(analysis)
        tool_span.set_output(sentiment)
    
    return sentiment
```

#### Mixing Sync and Async

You can use both patterns in the same application depending on your needs:

```python
class AIService:
    def __init__(self, client):
        self.client = client
    
    # Async method
    async def async_process(self, input_data):
        async with self.client.trace_model_call(...) as span:
            return await self._async_call(input_data)
    
    # Sync method
    def sync_process(self, input_data):
        with self.client.trace_model_call_sync(...) as span:
            return self._sync_call(input_data)
```

**When to use which:**
- Use **async** context managers (`trace_model_call`, `trace_tool_execution`) with `async`/`await` code
- Use **sync** context managers (`trace_model_call_sync`, `trace_tool_execution_sync`) with regular synchronous code
- The SDK handles timing, error tracking, and event submission automatically in both cases

---

### 4. Decorator Pattern

Decorators provide a clean, reusable way to add tracing to your functions without modifying their internal logic.

#### Model Call Decorators

```python
@client.trace_model_call_decorator(
    provider="openai",
    model="gpt-4-turbo"
)
async def generate_response(prompt: str) -> str:
    """This function is automatically traced"""
    response = await openai.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# Works with sync functions too
@client.trace_model_call_decorator(
    provider="anthropic",
    model="claude-3-sonnet"
)
def sync_generate(prompt: str) -> str:
    response = anthropic.messages.create(
        model="claude-3-sonnet-20240229",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text
```

#### Tool Execution Decorators

```python
@client.trace_tool_execution_decorator(
    tool_name="calculator",
    action="compute"
)
async def calculate(expression: str) -> float:
    """Automatically traced calculator tool"""
    return eval(expression)  # Don't use eval in production!

@client.trace_tool_execution_decorator(
    tool_name="web_scraper",
    action="fetch_page"
)
def scrape_website(url: str) -> str:
    response = requests.get(url)
    return response.text
```

#### Benefits of Decorators

1. **Clean Code**: Tracing logic is separated from business logic
2. **Reusable**: Apply the same tracing configuration to multiple functions
3. **Maintainable**: Easy to add or remove tracing without changing function internals
4. **Type-Safe**: Preserves function signatures and type hints

#### Advanced Decorator Usage

```python
# Combine multiple decorators
@client.trace_model_call_decorator(provider="openai", model="gpt-4")
@retry(max_attempts=3)
@cache(ttl=3600)
async def cached_llm_call(prompt: str) -> str:
    return await call_llm(prompt)

# Custom metadata
@client.trace_tool_execution_decorator(
    tool_name="database",
    action="query",
    meta_database="postgres",
    meta_query_type="select"
)
def query_db(sql: str) -> list:
    return db.execute(sql)
```

**Decorator Methods:**
- `trace_model_call_decorator(**kwargs)` - Trace model calls
- `trace_tool_execution_decorator(tool_name, **kwargs)` - Trace tool executions
- Both work with async and sync functions automatically

---

## 📊 Event Data and Metadata

### Standard Fields

Every traced event automatically captures:

- **Timing**: Start time, end time, latency
- **Status**: Success, error, or pending
- **Session Info**: Session ID, user ID, tenant ID
- **Component**: Source component identifier

### Custom Metadata

Add custom metadata to any trace for additional context:

```python
async with client.trace_model_call(
    provider="openai",
    model="gpt-4"
) as span:
    span.set_input(prompt)
    
    # Add custom metadata
    span.set_metadata("user_tier", "premium")
    span.set_metadata("feature_flag", "new_ui_v2")
    span.set_metadata("experiment_id", "exp_123")
    span.set_metadata("request_id", request.id)
    
    response = await call_model(prompt)
    span.set_output(response)
```

### Error Tracking

Errors are automatically captured and logged:

```python
async with client.trace_model_call(...) as span:
    try:
        span.set_input(prompt)
        response = await call_model(prompt)
        span.set_output(response)
    except Exception as e:
        # Error is automatically recorded
        span.set_metadata("error_type", type(e).__name__)
        span.set_metadata("error_details", str(e))
        raise
```

---

## 🔧 Configuration

### Client Configuration

```python
from telemetry_sdk.client import TelemetryClient

client = TelemetryClient(
    # Required
    api_key="your-api-key",
    

    application_id="my-app",
    
    # Session management
    session_id=None,  # Auto-generated if not provided
    
    # Performance tuning
    batch_size=10,  # Number of events to batch before sending
    flush_interval=5.0,  # Seconds between automatic flushes
    max_retries=3,  # Retry attempts for failed requests
    
    # Debugging
    debug=False  # Enable debug logging
)
```

### Environment Variables

You can also configure the SDK using environment variables:

```bash
export ROBOTFORGE_API_KEY="your-api-key"
```

```python
# SDK will automatically use environment variables
client = TelemetryClient()
```

---

## 🎓 Usage Examples

### Example 1: AI Chatbot

```python
from telemetry_sdk.client import TelemetryClient
import openai

class AIChatbot:
    def __init__(self, api_key: str, forge_key: str):
        self.openai = openai.AsyncOpenAI(api_key=api_key)
        self.forge = TelemetryClient(api_key=forge_key)
    
    async def chat(self, user_message: str, conversation_history: list) -> str:
        # Trace the model call
        async with self.forge.trace_model_call(
            provider="openai",
            model="gpt-4-turbo",
            temperature=0.7
        ) as span:
            # Prepare the prompt
            messages = conversation_history + [
                {"role": "user", "content": user_message}
            ]
            span.set_input(str(messages))
            
            # Call the model
            response = await self.openai.chat.completions.create(
                model="gpt-4-turbo",
                messages=messages,
                temperature=0.7
            )
            
            # Extract response
            assistant_message = response.choices[0].message.content
            
            # Record telemetry
            span.set_output(assistant_message)
            span.set_tokens(response.usage.total_tokens)
            span.set_metadata("conversation_length", len(messages))
            
            return assistant_message

# Usage
bot = AIChatbot(
    api_key="openai_key",
    forge_key="forge_key"
)

response = await bot.chat(
    "What's the weather like?",
    conversation_history=[]
)
```



### Example 3: Batch Processing

```python
from telemetry_sdk.client import TelemetryClient
import asyncio

async def process_documents(documents: list[str], forge_key: str):
    client = TelemetryClient(api_key=forge_key)
    
    async def process_one(doc: str):
        async with client.trace_model_call(
            provider="openai",
            model="gpt-4"
        ) as span:
            span.set_input(doc)
            span.set_metadata("doc_length", len(doc))
            
            summary = await summarize(doc)
            
            span.set_output(summary)
            span.set_metadata("summary_length", len(summary))
            
            return summary
    
    # Process all documents in parallel
    summaries = await asyncio.gather(*[
        process_one(doc) for doc in documents
    ])
    
    # Ensure all events are sent
    await client.flush()
    
    return summaries

# Usage
docs = [doc1, doc2, doc3, ...]
summaries = await process_documents(docs, "forge_key")
```

---

## 🔒 Best Practices

### 1. Always Flush Before Exit

```python
async def main():
    client = TelemetryClient(api_key="key")
    
    try:
        # Your application logic
        await do_work(client)
    finally:
        # Ensure all events are sent
        await client.flush()

asyncio.run(main())
```

### 2. Use Context Managers

Context managers automatically handle timing and error tracking:

```python
# ✅ Good - Automatic timing and error handling
async with client.trace_model_call(...) as span:
    result = await call_model()
    span.set_output(result)

# ❌ Avoid - Manual event building is error-prone
builder = client.create_model_call_event()
start = time.time()
result = await call_model()
builder.set_latency_ms(int((time.time() - start) * 1000))
await builder.send()
```

### 3. Add Meaningful Metadata

```python
async with client.trace_model_call(...) as span:
    span.set_input(prompt)
    
    # Add context that will help with debugging
    span.set_metadata("user_tier", user.tier)
    span.set_metadata("feature_version", "v2")
    span.set_metadata("ab_test_variant", "control")
    
    response = await call_model(prompt)
    span.set_output(response)
```

### 4. Handle Errors Gracefully

```python
async with client.trace_model_call(...) as span:
    try:
        span.set_input(prompt)
        response = await call_model(prompt)
        span.set_output(response)
    except RateLimitError as e:
        span.set_metadata("error", "rate_limit")
        span.set_metadata("retry_after", e.retry_after)
        raise
    except Exception as e:
        span.set_metadata("error", str(e))
        raise
```

### 5. Use Decorators for Reusable Functions

```python
# ✅ Good - Clean and reusable
@client.trace_model_call_decorator(provider="openai", model="gpt-4")
async def generate_text(prompt: str) -> str:
    return await call_openai(prompt)

# ❌ Avoid - Repeated tracing code
async def generate_text(prompt: str) -> str:
    async with client.trace_model_call(...) as span:
        span.set_input(prompt)
        result = await call_openai(prompt)
        span.set_output(result)
        return result
```

---

## 🚦 Coming Soon

We're actively developing additional features to make the RobotForge SDK even more powerful:

### Agent Support (Coming Q4 2025)
- **Agent Action Tracing**: Full support for tracing complex agent behaviors, planning, and reasoning
- **Multi-Agent Coordination**: Track interactions between multiple AI agents
- **Decision Trees**: Visualize agent decision-making processes

### Auto-Instrumentation (Coming Q4 2025)
- **Zero-Code Setup**: Automatically instrument popular AI frameworks
- **Supported Libraries**: 
  - OpenAI (GPT models)
  - Anthropic (Claude models)
  - LangChain
  - LlamaIndex
  - Hugging Face Transformers
- **Drop-in Integration**: Single line of code to enable comprehensive tracing

### Integrations (Coming Q1 2026)
- **Vector Databases**: Pinecone, Weaviate, Qdrant
- **Frameworks**: FastAPI, Flask, Django middleware
- **Observability Platforms**: OpenTelemetry, Datadog, New Relic
- **Cloud Platforms**: AWS Bedrock, Google Vertex AI, Azure OpenAI

### Watch Our Roadmap

We'll be publishing our detailed roadmap soon with dates, features, and opportunities for community input. Stay tuned to:

- **GitHub**: [github.com/robotforge/python-sdk](https://github.com/robotforge/python-sdk)



Want to influence our roadmap? Join our community and share your feedback!

---


## 📚 API Reference

### RobotForge Client

```python
class RobotForge:
    def __init__(
        self,
        api_key: str,
        tenant_id: str = None,
        application_id: str = None,
        session_id: str = None,
        batch_size: int = 10,
        flush_interval: float = 5.0,
        max_retries: int = 3,
        debug: bool = False
    )
```

### Context Managers

**Async:**
- `trace_model_call(**kwargs)` → AsyncContextManager[Span]
- `trace_tool_execution(tool_name: str, **kwargs)` → AsyncContextManager[Span]

**Sync:**
- `trace_model_call_sync(**kwargs)` → ContextManager[Span]
- `trace_tool_execution_sync(tool_name: str, **kwargs)` → ContextManager[Span]

### Decorators

- `trace_model_call_decorator(**kwargs)` → Callable
- `trace_tool_execution_decorator(tool_name: str, **kwargs)` → Callable

### Span Methods

**Common:**
- `set_input(text: str)` - Set input data
- `set_output(text: str)` - Set output data
- `set_metadata(key: str, value: Any)` - Add custom metadata
- `set_status(status: EventStatus)` - Set event status

**Model Call Specific:**
- `set_provider(name: str)` - Set model provider
- `set_model(name: str)` - Set model name
- `set_tokens(count: int)` - Set token count
- `set_cost(amount: float)` - Set cost
- `set_temperature(temp: float)` - Set temperature

**Tool Execution Specific:**
- `set_action(name: str)` - Set action name
- `set_endpoint(url: str)` - Set API endpoint
- `set_http_method(method: str)` - Set HTTP method
- `set_http_status_code(code: int)` - Set status code
- `set_request_payload(data: dict)` - Set request data
- `set_response_payload(data: dict)` - Set response data

### Utility Methods

- `flush() → None` - Send all pending events immediately
- `close() → None` - Close client and cleanup resources

---

## 🤝 Support

### Documentation
- **Full Docs**: [docs.robotforge.com.ng](https://docs.robotforge.com.ng)
- **API Reference**: [docs.robotforge.com.ng/api](https://docs.robotforge.com.ng/api)
- **Examples**: [github.com/robotforge/examples](https://github.com/robotforge/examples)

### Community
- **GitHub Issues**: [github.com/robotforge/python-sdk/issues](https://github.com/robotforge/python-sdk/issues)
- **Discord**: [discord.gg/robotforge](https://discord.gg/robotforge)
- **Stack Overflow**: Tag your questions with `robotforge`

### Enterprise Support
For enterprise support, SLAs, and custom solutions:
- **Email**: enterprise@robotforge.com.ng
- **Schedule a Call**: [robotforge.com.ng/contact](https://robotforge.com.ng/contact)

---

## 📄 License

The RobotForge Python SDK is released under the MIT License. See [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

Built with ❤️ by the RobotForge team. Special thanks to our early adopters and community contributors who helped shape this SDK.


*Version 1.0.0 | Last Updated: 29 October 2025*