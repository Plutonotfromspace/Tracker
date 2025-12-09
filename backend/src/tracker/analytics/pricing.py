"""
Pricing constants for xAI Grok API models.

This module contains the pricing information for different Grok models
used in the classification pipeline. Prices are per 1 million tokens.

Pricing source: https://docs.x.ai/docs/models
Last updated: December 2025
"""

from typing import Dict, Optional

# Pricing per 1 MILLION tokens (in USD)
# Format: {"input": price, "output": price}
PRICING: Dict[str, Dict[str, float]] = {
    # Grok 4.1 Fast models
    "grok-4-1-fast-reasoning": {"input": 0.20, "output": 0.50},
    "grok-4-1-fast-non-reasoning": {"input": 0.20, "output": 0.50},
    
    # Grok 4 Fast models (what we currently use)
    "grok-4-fast-reasoning": {"input": 0.20, "output": 0.50},
    "grok-4-fast-non-reasoning": {"input": 0.20, "output": 0.50},
    
    # Grok Code model
    "grok-code-fast-1": {"input": 0.20, "output": 1.50},
    
    # Grok 4 (full)
    "grok-4-0709": {"input": 3.00, "output": 15.00},
    "grok-4": {"input": 3.00, "output": 15.00},
    
    # Grok 3 models
    "grok-3-mini": {"input": 0.30, "output": 0.50},
    "grok-3": {"input": 3.00, "output": 15.00},
    
    # Grok 2 Vision
    "grok-2-vision-1212": {"input": 2.00, "output": 10.00},
}

# Default pricing for unknown models (conservative estimate)
DEFAULT_PRICING = {"input": 1.00, "output": 5.00}


def get_model_pricing(model: str) -> Dict[str, float]:
    """
    Get pricing for a specific model.
    
    Args:
        model: Model name (e.g., "grok-4-fast-reasoning")
    
    Returns:
        Dictionary with "input" and "output" prices per 1M tokens
    """
    # Try exact match first
    if model in PRICING:
        return PRICING[model]
    
    # Try partial match (for aliases like "grok-4-fast")
    for known_model, pricing in PRICING.items():
        if model in known_model or known_model in model:
            return pricing
    
    # Return default if unknown
    return DEFAULT_PRICING


def calculate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    image_tokens: int = 0,
    cached_tokens: int = 0
) -> float:
    """
    Calculate the cost of an API call.
    
    Args:
        model: Model name
        input_tokens: Number of input/prompt tokens
        output_tokens: Number of output/completion tokens
        image_tokens: Number of image tokens (charged as input)
        cached_tokens: Number of cached tokens (50% discount typically)
    
    Returns:
        Cost in USD
    """
    pricing = get_model_pricing(model)
    
    # Image tokens are charged at input rate
    total_input_tokens = input_tokens + image_tokens
    
    # Cached tokens get 50% discount (approximate)
    effective_input_tokens = total_input_tokens - (cached_tokens * 0.5)
    effective_input_tokens = max(0, effective_input_tokens)
    
    # Calculate costs (prices are per 1M tokens)
    input_cost = (effective_input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    
    return input_cost + output_cost


def format_cost(cost: float) -> str:
    """Format cost for display."""
    if cost < 0.01:
        return f"${cost:.6f}"
    elif cost < 1.00:
        return f"${cost:.4f}"
    else:
        return f"${cost:.2f}"
