"""
Setup configuration for the Telemetry SDK package
"""

from setuptools import setup, find_packages
from pathlib import Path

VERSION = "0.1.5"

# Read long description from README
def get_long_description():
    """Get long description from README.md"""
    readme_file = Path(__file__).parent / "README.md"
    try:
        with open(readme_file, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "Multi-layered Python SDK for AI/ML telemetry with support for context managers, decorators, auto-instrumentation, and logging integration."

# Read requirements
def read_requirements(filename):
    """Read requirements from file"""
    req_file = Path(__file__).parent / filename
    try:
        with open(req_file, "r", encoding="utf-8") as f:
            return [
                line.strip() 
                for line in f 
                if line.strip() and not line.startswith("#")
            ]
    except FileNotFoundError:
        return []

setup(
    name="robotforge-python-sdk",
    version=VERSION,
    author="RobotForge",
    author_email="support@robotforge.com.ng",
    description="Multi-layered Python SDK for AI/ML telemetry",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/robotforge/telemetry-sdk",
    project_urls={
        "Bug Reports": "https://github.com/robotforge/python-sdk/issues",
        "Source": "https://github.com/robotforge/python-sdk",
        "Documentation": "https://robotforge-python-sdk.readthedocs.io/",
        "Changelog": "https://github.com/robotforge/python-sdk/blob/main/CHANGELOG.md",
    },
    packages=find_packages(exclude=["tests*", "docs*", "examples*"]),
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
        "Typing :: Typed",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements("requirements.txt") or [
        "aiohttp>=3.8.0",
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
            "httpx>=0.24.0",
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
            "pytest-mock>=3.10.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
            "pre-commit>=2.20.0",
            "twine>=4.0.0",
            "build>=0.10.0",
        ],
        
        # Documentation
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.2.0",
            "sphinxcontrib-asyncio>=0.3.0",
            "myst-parser>=0.18.0",
        ],
        
        # Configuration file support
        "config": [
            "PyYAML>=6.0",
            "toml>=0.10.0",
        ],
        
        # All optional dependencies
        "all": [
            "openai>=1.0.0",
            "anthropic>=0.25.0", 
            "langchain>=0.1.0",
            "llama-index>=0.9.0",
            "requests>=2.28.0",
            "httpx>=0.24.0",
            "fastapi>=0.100.0",
            "flask>=2.0.0",
            "PyYAML>=6.0",
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
            "*.yaml",
            "*.json",
        ],
    },
    keywords=[
        "telemetry", 
        "monitoring", 
        "ai", 
        "ml", 
        "llm", 
        "observability", 
        "tracing", 
        "logging",
        "openai",
        "anthropic",
        "langchain",
        "instrumentation"
    ],
    zip_safe=False,  # For better compatibility
)