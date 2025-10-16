"""
LLM pricing database for cost calculation.
Prices are per 1 million tokens unless otherwise specified.
"""

from typing import Dict, Optional, Tuple


class LLMPricing:
    """Database of LLM pricing for cost calculations"""
    
    # Pricing structure: provider -> model -> (prompt_price_per_1m, completion_price_per_1m)
    PRICING: Dict[str, Dict[str, Tuple[float, float]]] = {
        'openai': {
            # GPT-4 models
            'gpt-4': (30.0, 60.0),
            'gpt-4-0613': (30.0, 60.0),
            'gpt-4-32k': (60.0, 120.0),
            'gpt-4-32k-0613': (60.0, 120.0),
            'gpt-4-turbo': (10.0, 30.0),
            'gpt-4-turbo-preview': (10.0, 30.0),
            'gpt-4-1106-preview': (10.0, 30.0),
            'gpt-4-0125-preview': (10.0, 30.0),
            'gpt-4-turbo-2024-04-09': (10.0, 30.0),
            'gpt-4o': (5.0, 15.0),
            'gpt-4o-2024-05-13': (5.0, 15.0),
            'gpt-4o-2024-08-06': (2.5, 10.0),
            'gpt-4o-mini': (0.15, 0.6),
            'gpt-4o-mini-2024-07-18': (0.15, 0.6),
            
            # GPT-3.5 models
            'gpt-3.5-turbo': (0.5, 1.5),
            'gpt-3.5-turbo-0125': (0.5, 1.5),
            'gpt-3.5-turbo-1106': (1.0, 2.0),
            'gpt-3.5-turbo-16k': (3.0, 4.0),
            'gpt-3.5-turbo-instruct': (1.5, 2.0),
            
            # O1 models
            'o1-preview': (15.0, 60.0),
            'o1-preview-2024-09-12': (15.0, 60.0),
            'o1-mini': (3.0, 12.0),
            'o1-mini-2024-09-12': (3.0, 12.0),
            
            # Embedding models (input only)
            'text-embedding-3-small': (0.02, 0.0),
            'text-embedding-3-large': (0.13, 0.0),
            'text-embedding-ada-002': (0.10, 0.0),
        },
        
        'anthropic': {
            # Claude 3.5 models
            'claude-3-5-sonnet-20241022': (3.0, 15.0),
            'claude-3-5-sonnet-20240620': (3.0, 15.0),
            'claude-3-5-haiku-20241022': (1.0, 5.0),
            
            # Claude 3 models
            'claude-3-opus-20240229': (15.0, 75.0),
            'claude-3-sonnet-20240229': (3.0, 15.0),
            'claude-3-haiku-20240307': (0.25, 1.25),
            
            # Claude 2 models
            'claude-2.1': (8.0, 24.0),
            'claude-2.0': (8.0, 24.0),
            'claude-instant-1.2': (0.8, 2.4),
        },
        
        'google': {
            # Gemini models
            'gemini-1.5-pro': (1.25, 5.0),
            'gemini-1.5-pro-001': (1.25, 5.0),
            'gemini-1.5-pro-002': (1.25, 5.0),
            'gemini-1.5-flash': (0.075, 0.30),
            'gemini-1.5-flash-001': (0.075, 0.30),
            'gemini-1.5-flash-002': (0.075, 0.30),
            'gemini-1.0-pro': (0.5, 1.5),
            'gemini-pro': (0.5, 1.5),
            
            # Embedding models
            'text-embedding-004': (0.00001, 0.0),  # $0.00001 per 1k characters
        },
        
        'cohere': {
            # Command models
            'command': (1.0, 2.0),
            'command-light': (0.3, 0.6),
            'command-nightly': (1.0, 2.0),
            'command-light-nightly': (0.3, 0.6),
            'command-r': (0.5, 1.5),
            'command-r-plus': (3.0, 15.0),
            
            # Embedding models
            'embed-english-v3.0': (0.1, 0.0),
            'embed-multilingual-v3.0': (0.1, 0.0),
        },
        
        'mistral': {
            # Mistral models
            'mistral-large-latest': (4.0, 12.0),
            'mistral-medium-latest': (2.7, 8.1),
            'mistral-small-latest': (1.0, 3.0),
            'mistral-tiny': (0.25, 0.75),
            'open-mistral-7b': (0.25, 0.25),
            'open-mixtral-8x7b': (0.7, 0.7),
            'open-mixtral-8x22b': (2.0, 6.0),
        },
        
        'together': {
            # Meta Llama models
            'meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo': (1.2, 1.2),
            'meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo': (3.5, 3.5),
            'meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo': (0.88, 0.88),
            'meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo': (0.18, 0.18),
            
            # Mixtral models
            'mistralai/Mixtral-8x7B-Instruct-v0.1': (0.6, 0.6),
            'mistralai/Mixtral-8x22B-Instruct-v0.1': (1.2, 1.2),
        },
    }
    
    @classmethod
    def calculate_cost(
        cls,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int
    ) -> Optional[float]:
        """
        Calculate the cost of an LLM call.
        
        Args:
            provider: Provider name (e.g., 'openai', 'anthropic')
            model: Model name (e.g., 'gpt-4', 'claude-3-opus-20240229')
            prompt_tokens: Number of input/prompt tokens
            completion_tokens: Number of output/completion tokens
        
        Returns:
            Total cost in USD, or None if pricing not available
        """
        provider = provider.lower() if provider else ''
        model = model.lower() if model else ''
        
        # Try exact match first
        if provider in cls.PRICING and model in cls.PRICING[provider]:
            prompt_price, completion_price = cls.PRICING[provider][model]
            cost = (
                (prompt_tokens * prompt_price / 1_000_000) +
                (completion_tokens * completion_price / 1_000_000)
            )
            return round(cost, 6)
        
        # Try partial match for model name (handles versioned models)
        if provider in cls.PRICING:
            for known_model, (prompt_price, completion_price) in cls.PRICING[provider].items():
                if known_model in model or model in known_model:
                    cost = (
                        (prompt_tokens * prompt_price / 1_000_000) +
                        (completion_tokens * completion_price / 1_000_000)
                    )
                    return round(cost, 6)
        
        return None
    
    @classmethod
    def get_model_pricing(
        cls,
        provider: str,
        model: str
    ) -> Optional[Tuple[float, float]]:
        """
        Get the pricing for a specific model.
        
        Args:
            provider: Provider name
            model: Model name
        
        Returns:
            Tuple of (prompt_price_per_1m, completion_price_per_1m) or None
        """
        provider = provider.lower() if provider else ''
        model = model.lower() if model else ''
        
        if provider in cls.PRICING and model in cls.PRICING[provider]:
            return cls.PRICING[provider][model]
        
        return None
    
    @classmethod
    def add_custom_pricing(
        cls,
        provider: str,
        model: str,
        prompt_price_per_1m: float,
        completion_price_per_1m: float
    ) -> None:
        """
        Add custom pricing for a provider/model combination.
        
        Args:
            provider: Provider name
            model: Model name
            prompt_price_per_1m: Price per 1M prompt tokens
            completion_price_per_1m: Price per 1M completion tokens
        """
        provider = provider.lower()
        model = model.lower()
        
        if provider not in cls.PRICING:
            cls.PRICING[provider] = {}
        
        cls.PRICING[provider][model] = (prompt_price_per_1m, completion_price_per_1m)