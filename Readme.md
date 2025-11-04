Perfect! Here's a **ready-to-publish GitHub-friendly Markdown README** for RobotForge Python SDK, with collapsible code sections, badges, and a clean structure:

---

# ![RobotForge Logo](https://robotforge.com.ng/wp-content/uploads/2025/07/ChatGPT-Image-Jul-26-2025-02_22_37-PM.png) 
# RobotForge Python SDK

[![PyPI](https://img.shields.io/pypi/v/robotforge-python-sdk)](https://pypi.org/project/robotforge-python-sdk/)
[![Python Version](https://img.shields.io/pypi/pyversions/robotforge-python-sdk)](https://pypi.org/project/robotforge-python-sdk/)
[![License](https://img.shields.io/github/license/robotforge/python-sdk)](LICENSE)
[![Discord](https://img.shields.io/discord/123456789)](https://discord.gg/robotforge)

**Build trust and reliability into your AI applications with comprehensive telemetry and observability.**

---

## 🎯 Introduction

The **RobotForge Python SDK** provides full observability into AI systems—**model calls, tool executions, agent behaviors**—enabling debugging, performance optimization, and trust.

It works with **any LLM or AI tool**; OpenAI is used only as an example.

**Why Use RobotForge?**

* **Debug with Confidence** – Trace the full chain of model calls and tool executions.
* **Optimize Performance** – Track tokens, latency, and cost for informed decisions.
* **Build Trust** – Ensure transparency for auditing, compliance, and stakeholders.
* **Monitor Production** – Detect errors and anomalies in real time.
* **Understand Usage** – Analyze user behavior and AI tool usage patterns.

---

## 📦 Installation

```bash
pip install robotforge-python-sdk
```

**Requirements:**

* Python 3.8+
* `aiohttp` for async operations
* `requests` for sync operations

---

## 🚀 Quick Start

<details>
<summary>Basic Setup</summary>

```python
import time
import os
import asyncio
from dotenv import load_dotenv
import telemetry_sdk

load_dotenv()

API_KEY = os.getenv("ROBOTFORGE_API_KEY")
APPLICATION_ID = "example_app"

client = telemetry_sdk.quick_setup(
    api_key=API_KEY,
    application_id=APPLICATION_ID,
    set_as_default=True
)
```

</details>

<details>
<summary>Async Model Call Example</summary>

```python
async def call_model(prompt: str):
    async with client.trace_model_call(provider="example_llm", model="custom-model") as span:
        span.set_input(prompt)
        
        # Replace with your actual LLM call
        response = await some_llm_client.generate(prompt)
        
        span.set_output(response)
        span.set_tokens(response.usage.total_tokens)
        return response
```

</details>

<details>
<summary>Sync Model Call Example</summary>

```python
with client.trace_model_call_sync(provider="example_llm", model="custom-model") as span:
    span.set_input(prompt)
    response = some_llm_client.generate_sync(prompt)
    span.set_output(response)
    span.set_tokens(response.usage.total_tokens)
```

</details>

<details>
<summary>Decorator Example</summary>

```python
@client.trace_model_call_decorator(provider="example_llm", model="custom-model")
async def summarize_text(text: str):
    response = await some_llm_client.generate(text)
    return response
```

</details>

---

## 🏗️ Core Features

### 1. Model Call Tracing

* Async: `trace_model_call(...)` → `AsyncContextManager[Span]`
* Sync: `trace_model_call_sync(...)` → `ContextManager[Span]`
* Decorator: `trace_model_call_decorator(...)` → Callable

### 2. Tool Execution Tracing

* Async: `trace_tool_execution(...)` → `AsyncContextManager[Span]`
* Sync: `trace_tool_execution_sync(...)` → `ContextManager[Span]`
* Decorator: `trace_tool_execution_decorator(tool_name, ...)` → Callable

### 3. Flexible Context Managers

Supports **sync and async environments**, or mixing both.

```python
# Async model + tool
async with client.trace_model_call(provider="example_llm") as m_span:
    response = await llm.generate(prompt)
    m_span.set_output(response)

async with client.trace_tool_execution(tool_name="api_tool") as t_span:
    result = await api_tool.call()
    t_span.set_output(result)
```

```python
# Sync model + tool
with client.trace_model_call_sync(provider="example_llm") as m_span:
    response = llm.generate_sync(prompt)
    m_span.set_output(response)

with client.trace_tool_execution_sync(tool_name="api_tool") as t_span:
    result = api_tool.call_sync()
    t_span.set_output(result)
```

### 4. Full Example: Async Tracing

```python
import asyncio

prompts = [
    "Write about sunshine in one sentence.",
    "Write about apples in one sentence.",
    "Write about technology in one sentence."
]

async def process_prompt(prompt: str):
    async with client.trace_model_call(provider="example_llm", model="custom-model") as span:
        span.set_input(prompt)
        response = await some_llm_client.generate(prompt)
        span.set_output(response)
        span.set_tokens(response.usage.total_tokens)
        return response

results = await asyncio.gather(*[process_prompt(p) for p in prompts])
await client.flush()
```

---

## 🔧 Configuration

```python
from telemetry_sdk.client import TelemetryClient

client = TelemetryClient(
    api_key="your-api-key",
    application_id="my_app",
    session_id=None,      # auto-generated
    batch_size=10,
    flush_interval=5.0,
    max_retries=3,
    debug=False
)
```

Environment variable support:

```bash
export ROBOTFORGE_API_KEY="your-api-key"
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE)

---

## 🤝 Support & Community

* **Docs:** [docs.robotforge.com.ng](https://github.com/robotforge/Readmd.md)
* **Examples:** [github.com/robotforge/examples](https://github.com/robotforge/examples)


---

*Version 1.0.1 | Last Updated: 4 November 2025*


