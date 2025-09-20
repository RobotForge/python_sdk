"""
Setup configuration for the Telemetry SDK package
"""

from setuptools import setup, find_packages

# Read long description from README
try:
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()
except FileNotFoundError:
    long_description = "Multi-layered Python SDK for AI/ML telemetry with support for context managers, decorators, auto-instrumentation, and logging integration."

# Read requirements
def read_requirements(filename):
    try:
        with open(filename, "r") as f:
            return [line.strip() for line in f if line.strip() and not line.startswith("#")]
    except FileNotFoundError:
        return []

setup(
    name="telemetry-sdk",
    version="1.0.0",
    author="Your Organization",
    author_email="support@yourorg.com",
    description="Multi-layered Python SDK for AI/ML telemetry",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourorg/telemetry-sdk",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: System :: Monitoring",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "aiohttp>=3.8.0",
        "asyncio-mqtt>=0.13.0",  # For potential MQTT support
        "pydantic>=1.10.0",
        "typing-extensions>=4.0.0",
    ],
    extras_require={
        # Core auto-instrumentation dependencies
        "auto": [
            "openai>=1.0.0",
            "anthropic>=0.25.0",
            "langchain>=0.1.0",
            "llama-index>=0.9.0",
            "requests>=2.28.0",
        ],
        
        # Web framework integrations
        "web": [
            "fastapi>=0.100.0",
            "flask>=2.0.0",
            "django>=4.0.0",
        ],
        
        # Additional ML/AI libraries
        "ml": [
            "transformers>=4.20.0",
            "torch>=1.12.0",
            "tensorflow>=2.10.0",
            "scikit-learn>=1.1.0",
            "numpy>=1.21.0",
            "pandas>=1.4.0",
        ],
        
        # Development dependencies
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
            "pre-commit>=2.20.0",
        ],
        
        # Documentation
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.2.0",
            "sphinxcontrib-asyncio>=0.3.0",
        ],
        
        # All optional dependencies
        "all": [
            "openai>=1.0.0",
            "anthropic>=0.25.0", 
            "langchain>=0.1.0",
            "llama-index>=0.9.0",
            "requests>=2.28.0",
            "fastapi>=0.100.0",
            "flask>=2.0.0",
            "transformers>=4.20.0",
            "torch>=1.12.0",
            "numpy>=1.21.0",
            "pandas>=1.4.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "telemetry-cli=telemetry_sdk.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "telemetry_sdk": [
            "py.typed",  # For type checking support
            "config/*.yaml",
            "config/*.json",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/yourorg/telemetry-sdk/issues",
        "Source": "https://github.com/yourorg/telemetry-sdk",
        "Documentation": "https://telemetry-sdk.readthedocs.io/",
    },
    keywords="telemetry, monitoring, ai, ml, llm, observability, tracing, logging",
)

# Additional setup.cfg content
SETUP_CFG = """
[metadata]
name = telemetry-sdk
version = attr: telemetry_sdk.__version__
description = Multi-layered Python SDK for AI/ML telemetry
long_description = file: README.md
long_description_content_type = text/markdown
url = https://github.com/yourorg/telemetry-sdk
author = Your Organization
author_email = support@yourorg.com
license = MIT
license_files = LICENSE
classifiers =
    Development Status :: 4 - Beta
    Intended Audience :: Developers
    License :: OSI Approved :: MIT License
    Operating System :: OS Independent
    Programming Language :: Python :: 3
    Programming Language :: Python :: 3.8
    Programming Language :: Python :: 3.9
    Programming Language :: Python :: 3.10
    Programming Language :: Python :: 3.11
    Programming Language :: Python :: 3.12

[options]
packages = find:
python_requires = >=3.8
install_requires =
    aiohttp>=3.8.0
    pydantic>=1.10.0
    typing-extensions>=4.0.0

[options.packages.find]
exclude =
    tests*
    docs*
    examples*

[options.extras_require]
auto = 
    openai>=1.0.0
    anthropic>=0.25.0
    langchain>=0.1.0
    llama-index>=0.9.0
    requests>=2.28.0

dev =
    pytest>=7.0.0
    pytest-asyncio>=0.21.0
    pytest-cov>=4.0.0
    black>=22.0.0
    isort>=5.10.0
    flake8>=5.0.0
    mypy>=1.0.0

all =
    %(auto)s
    fastapi>=0.100.0
    flask>=2.0.0
    transformers>=4.20.0
    
[options.entry_points]
console_scripts =
    telemetry-cli = telemetry_sdk.cli:main

[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    -v
    --tb=short
    --strict-markers
    --asyncio-mode=auto
markers =
    slow: marks tests as slow
    integration: marks tests as integration tests
    unit: marks tests as unit tests

[coverage:run]
source = telemetry_sdk
omit = 
    */tests/*
    */test_*
    setup.py

[coverage:report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise AssertionError
    raise NotImplementedError
    if __name__ == .__main__.:

[mypy]
python_version = 3.8
warn_return_any = True
warn_unused_configs = True
disallow_untyped_defs = True

[isort]
profile = black
multi_line_output = 3
line_length = 88
known_first_party = telemetry_sdk

[flake8]
max-line-length = 88
extend-ignore = E203, W503
"""

# pyproject.toml content
PYPROJECT_TOML = """
[build-system]
requires = ["setuptools>=45", "wheel", "setuptools_scm[toml]>=6.2"]
build-backend = "setuptools.build_meta"

[project]
name = "telemetry-sdk"
dynamic = ["version"]
description = "Multi-layered Python SDK for AI/ML telemetry"
readme = "README.md"
license = {file = "LICENSE"}
authors = [
    {name = "Your Organization", email = "support@yourorg.com"},
]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Software Development :: Libraries :: Python Modules",
    "Topic :: System :: Monitoring",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]
requires-python = ">=3.8"
dependencies = [
    "aiohttp>=3.8.0",
    "pydantic>=1.10.0",
    "typing-extensions>=4.0.0",
]

[project.optional-dependencies]
auto = [
    "openai>=1.0.0",
    "anthropic>=0.25.0",
    "langchain>=0.1.0",
    "llama-index>=0.9.0",
    "requests>=2.28.0",
]
web = [
    "fastapi>=0.100.0",
    "flask>=2.0.0",
    "django>=4.0.0",
]
ml = [
    "transformers>=4.20.0",
    "torch>=1.12.0",
    "tensorflow>=2.10.0",
    "scikit-learn>=1.1.0",
    "numpy>=1.21.0",
    "pandas>=1.4.0",
]
dev = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
    "pytest-cov>=4.0.0",
    "black>=22.0.0",
    "isort>=5.10.0",
    "flake8>=5.0.0",
    "mypy>=1.0.0",
    "pre-commit>=2.20.0",
]
docs = [
    "sphinx>=5.0.0",
    "sphinx-rtd-theme>=1.2.0",
    "sphinxcontrib-asyncio>=0.3.0",
]
all = [
    "telemetry-sdk[auto,web,ml]",
]

[project.urls]
Homepage = "https://github.com/yourorg/telemetry-sdk"
Documentation = "https://telemetry-sdk.readthedocs.io/"
Repository = "https://github.com/yourorg/telemetry-sdk.git"
"Bug Tracker" = "https://github.com/yourorg/telemetry-sdk/issues"

[project.scripts]
telemetry-cli = "telemetry_sdk.cli:main"

[tool.setuptools]
packages = ["telemetry_sdk"]

[tool.setuptools.dynamic]
version = {attr = "telemetry_sdk.__version__"}

[tool.black]
line-length = 88
target-version = ['py38']
include = '\.pyi?
extend-exclude = '''
/(
  # directories
  \.eggs
  | \.git
  | \.hg
  | \.mypy_cache
  | \.tox
  | \.venv
  | build
  | dist
)/
'''

[tool.isort]
profile = "black"
multi_line_output = 3
line_length = 88
known_first_party = ["telemetry_sdk"]

[tool.mypy]
python_version = "3.8"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
ignore_missing_imports = true

[[tool.mypy.overrides]]
module = "tests.*"
ignore_errors = true

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
python_classes = "Test*"
python_functions = "test_*"
addopts = [
    "-v",
    "--tb=short", 
    "--strict-markers",
    "--asyncio-mode=auto"
]
markers = [
    "slow: marks tests as slow",
    "integration: marks tests as integration tests",
    "unit: marks tests as unit tests",
]

[tool.coverage.run]
source = ["telemetry_sdk"]
omit = [
    "*/tests/*",
    "*/test_*",
    "setup.py",
]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise AssertionError", 
    "raise NotImplementedError",
    "if __name__ == .__main__.:",
]
"""

# Requirements files content
REQUIREMENTS_TXT = """
# Core dependencies
aiohttp>=3.8.0
pydantic>=1.10.0
typing-extensions>=4.0.0
"""

REQUIREMENTS_DEV_TXT = """
# Development dependencies
-r requirements.txt

# Testing
pytest>=7.0.0
pytest-asyncio>=0.21.0
pytest-cov>=4.0.0
pytest-mock>=3.10.0

# Code formatting and linting
black>=22.0.0
isort>=5.10.0
flake8>=5.0.0
mypy>=1.0.0
pre-commit>=2.20.0

# Documentation
sphinx>=5.0.0
sphinx-rtd-theme>=1.2.0
sphinxcontrib-asyncio>=0.3.0

# Auto-instrumentation testing
openai>=1.0.0
anthropic>=0.25.0
langchain>=0.1.0
requests>=2.28.0
"""

# Makefile content
MAKEFILE = """
.PHONY: help install install-dev test test-cov lint format clean build docs

help:
	@echo "Available commands:"
	@echo "  install     - Install package in development mode"
	@echo "  install-dev - Install with development dependencies"
	@echo "  test        - Run tests"
	@echo "  test-cov    - Run tests with coverage"
	@echo "  lint        - Run linting (flake8, mypy)"
	@echo "  format      - Format code (black, isort)"
	@echo "  clean       - Clean build artifacts"
	@echo "  build       - Build package"
	@echo "  docs        - Build documentation"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev,auto,all]"
	pre-commit install

test:
	pytest

test-cov:
	pytest --cov=telemetry_sdk --cov-report=html --cov-report=term

lint:
	flake8 telemetry_sdk tests
	mypy telemetry_sdk

format:
	black telemetry_sdk tests examples
	isort telemetry_sdk tests examples

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: clean
	python -m build

docs:
	cd docs && make html

publish: build
	python -m twine upload dist/*

publish-test: build
	python -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*
"""

# GitHub Actions workflow
GITHUB_WORKFLOW = """
name: CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, "3.10", "3.11", "3.12"]

    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Cache pip packages
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements*.txt') }}
        restore-keys: |
          ${{ runner.os }}-pip-
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev,auto]"
    
    - name: Lint with flake8
      run: |
        flake8 telemetry_sdk tests --count --select=E9,F63,F7,F82 --show-source --statistics
        flake8 telemetry_sdk tests --count --exit-zero --max-complexity=10 --max-line-length=88 --statistics
    
    - name: Type check with mypy
      run: |
        mypy telemetry_sdk
    
    - name: Test with pytest
      run: |
        pytest --cov=telemetry_sdk --cov-report=xml
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests
        name: codecov-umbrella

  build:
    needs: test
    runs-on: ubuntu-latest
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: "3.11"
    
    - name: Install build dependencies
      run: |
        python -m pip install --upgrade pip
        pip install build twine
    
    - name: Build package
      run: python -m build
    
    - name: Check package
      run: twine check dist/*
    
    - name: Upload build artifacts
      uses: actions/upload-artifact@v3
      with:
        name: dist
        path: dist/
"""

# Pre-commit configuration
PRE_COMMIT_CONFIG = """
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
      - id: check-merge-conflict
      - id: debug-statements

  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black
        language_version: python3.8

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
        args: ["--profile", "black"]

  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        args: [--max-line-length=88, --extend-ignore=E203,W503]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.3.0
    hooks:
      - id: mypy
        additional_dependencies: [types-requests]
        args: [--ignore-missing-imports]
"""

# README.md content
README_MD = '''
# Telemetry SDK

A comprehensive, multi-layered Python SDK for AI/ML telemetry that supports context managers, decorators, auto-instrumentation, and logging integration.

## 🚀 Features

- **Multiple Integration Patterns**: Choose from context managers, decorators, auto-instrumentation, or logging
- **Zero-Code Auto-Instrumentation**: Automatically trace OpenAI, Anthropic, LangChain, and other popular libraries
- **Async-First Design**: Built for modern async Python applications
- **Rich Event Types**: Support for model calls, tool executions, agent actions, and MCP events
- **Framework Integrations**: Built-in support for FastAPI, LangChain, LlamaIndex
- **Flexible Configuration**: Extensive customization options
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

logger.tool_execution(
    "Web search completed",
    tool_name="bing_search",
    latency_ms=500,
    http_status_code=200
)
```

## 🔧 Advanced Usage

### Real-World AI Agent Example

```python
class AIAgent:
    def __init__(self, telemetry_client):
        self.telemetry = telemetry_client
    
    async def process_request(self, user_input: str) -> str:
        async with self.telemetry.trace_agent_action(
            action_type="conversation",
            source_component="ai_agent"
        ) as conversation:
            conversation.set_input(user_input)
            
            # Step 1: Analyze intent
            intent = await self._analyze_intent(user_input)
            conversation.set_metadata("intent", intent)
            
            # Step 2: Execute actions
            if intent == "search":
                results = await self._web_search(user_input)
            elif intent == "calculate":
                results = await self._calculate(user_input)
            
            # Step 3: Generate response
            response = await self._generate_response(user_input, results)
            conversation.set_output(response)
            
            return response
    
    @telemetry.trace_model_call_decorator(provider="openai")
    async def _analyze_intent(self, text: str) -> str:
        # Automatically traced model call
        pass
    
    @telemetry.trace_tool_execution_decorator(tool_name="web_search") 
    async def _web_search(self, query: str):
        # Automatically traced tool execution
        pass
```

### Framework Integrations

```python
# FastAPI Integration
from fastapi import FastAPI
from telemetry_sdk import FrameworkIntegrations

app = FastAPI()
integrations = FrameworkIntegrations(client)
integrations.wrap_fastapi_app(app)  # All endpoints automatically traced

# LangChain Integration
chain = LLMChain(llm=OpenAI(), prompt=prompt_template)
traced_chain = integrations.wrap_langchain_chain(chain)
result = traced_chain.run(input="Hello")  # Automatically traced
```

## 📊 Event Types

The SDK supports four main event types:

- **Model Calls**: LLM API calls (OpenAI, Anthropic, etc.)
- **Tool Executions**: External API calls, database queries, etc.
- **Agent Actions**: Planning, reasoning, decision-making
- **MCP Events**: Model Context Protocol events

## 🛠️ Configuration

### Environment Variables

```bash
TELEMETRY_API_KEY=your-api-key
TELEMETRY_ENDPOINT=https://your-server.com
TELEMETRY_PROJECT_ID=my-project
TELEMETRY_AUTO_SEND=true
TELEMETRY_BATCH_SIZE=50
```

### Programmatic Configuration

```python
client = TelemetryClient(
    api_key="your-key",
    endpoint="https://your-server.com",
    project_id="my-project",
    auto_send=True,           # Enable background batching
    batch_size=50,            # Events per batch
    batch_timeout=5.0,        # Max time between sends
    retry_attempts=3,         # Retry failed requests
    pii_scrubbing=True,       # Auto-scrub PII
    max_payload_size=100_000  # Max event size
)
```

## 🧪 Testing

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=telemetry_sdk --cov-report=html

# Run specific test types
pytest -m unit
pytest -m integration
```

## 📚 Documentation

- [Full Documentation](https://telemetry-sdk.readthedocs.io/)
- [API Reference](https://telemetry-sdk.readthedocs.io/en/latest/api.html)
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
- [Discord Community](https://discord.gg/your-server)

## 🗺️ Roadmap

- [ ] Additional auto-instrumentation targets (Hugging Face, etc.)
- [ ] GraphQL and gRPC support
- [ ] Real-time streaming capabilities
- [ ] Enhanced PII detection and scrubbing
- [ ] Distributed tracing integration (OpenTelemetry)
- [ ] Custom metric aggregations
'''