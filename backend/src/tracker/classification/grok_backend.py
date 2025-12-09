"""
Grok VLM backend for body type classification.

This module provides a cloud-based VLM classifier using xAI's Grok-4
for accurate body type classification through visual reasoning.

Unlike the local Ollama-based VLM backend, this uses xAI's cloud API
which provides better accuracy (78% MMMU vs 58.6% for local models)
at a very low cost ($0.35 per 1,000 images).

Requirements:
    - xai-sdk Python package: `pip install xai-sdk`
    - XAI_API_KEY environment variable set
"""

import asyncio
import base64
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from src.tracker.utils.logging_config import get_logger

logger = get_logger(__name__)

# Try to import analytics (may not be available during initial setup)
try:
    from src.tracker.analytics import get_cost_tracker, APICallRecord
    ANALYTICS_AVAILABLE = True
except ImportError:
    ANALYTICS_AVAILABLE = False

# Try to import xai_sdk
try:
    from xai_sdk import Client, AsyncClient
    from xai_sdk.chat import user, image
    XAI_AVAILABLE = True
except ImportError:
    XAI_AVAILABLE = False
    logger.warning("xai-sdk package not installed. Run: pip install xai-sdk")


# =============================================================================
# PYDANTIC SCHEMAS FOR STRUCTURED OUTPUT
# =============================================================================

class BodyTypeSchema(BaseModel):
    """Pydantic schema for body type classification structured output."""
    body_type: str = Field(
        description="Body type classification: 'reefer', 'dry_van', or 'flatbed'"
    )
    confidence: float = Field(
        description="Confidence score between 0.0 and 1.0",
        ge=0.0,
        le=1.0
    )
    reasoning: str = Field(
        description="Explanation of the classification decision"
    )


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class GrokConfig:
    """
    Configuration for the Grok backend.
    
    Attributes:
        model: Grok model name (default: grok-4-fast-reasoning)
        api_key: xAI API key (default: from XAI_API_KEY env var)
        temperature: Sampling temperature (0 = deterministic)
        detail: Image detail level ("low", "high", "auto")
        max_retries: Number of retries on failure
    """
    model: str = "grok-4-1-fast"  # Latest fast reasoning model with vision
    api_key: Optional[str] = None  # Will use XAI_API_KEY env var if not set
    temperature: float = 0.0  # Deterministic for classification
    detail: str = "high"  # High detail for reefer unit detection
    max_retries: int = 2
    
    def __post_init__(self):
        """Load API key from environment variable."""
        if self.api_key is None:
            self.api_key = os.getenv("XAI_API_KEY")
            if not self.api_key:
                raise ValueError("XAI_API_KEY environment variable must be set")


# =============================================================================
# RESULT DATACLASS
# =============================================================================

@dataclass
class GrokClassificationResult:
    """
    Result from Grok classification.
    
    Attributes:
        body_type: Classified body type (reefer, dry_van, flatbed)
        confidence: Confidence score (0-1)
        reasoning: Grok's explanation of its decision
        raw_response: Full response from Grok
        model: Model used for classification
    """
    body_type: str  # reefer, dry_van, flatbed
    confidence: float
    reasoning: str
    raw_response: Optional[str] = None
    model: str = ""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "body_type": self.body_type,
            "confidence": round(self.confidence, 4),
            "reasoning": self.reasoning,
            "model": self.model,
        }


# =============================================================================
# GROK BACKEND
# =============================================================================

class GrokBackend:
    """
    Grok backend for body type classification using xAI's API.
    
    This backend uses Grok-4 vision models that score 78% on MMMU
    benchmarks, significantly better than local 7B models (58.6%).
    
    The cloud API approach provides:
    1. Better accuracy for detecting refrigeration units
    2. No local GPU requirements
    3. Very low cost ($0.35 per 1,000 images)
    
    Example:
        >>> backend = GrokBackend()
        >>> result = backend.classify("truck_image.jpg")
        >>> print(f"{result.body_type}: {result.confidence:.0%}")
        >>> print(f"Reasoning: {result.reasoning}")
        reefer: 95%
        Reasoning: I can see a white refrigeration unit mounted on the front of the trailer.
    """
    
    def __init__(self, config: Optional[GrokConfig] = None):
        """
        Initialize the Grok backend.
        
        Args:
            config: Grok configuration. Uses defaults if None.
        
        Raises:
            ImportError: If xai-sdk package is not installed.
            ValueError: If XAI_API_KEY is not set.
        """
        if not XAI_AVAILABLE:
            raise ImportError(
                "xai-sdk package not installed. "
                "Install with: pip install xai-sdk"
            )
        
        self.config = config or GrokConfig()
        
        if not self.config.api_key:
            raise ValueError(
                "XAI_API_KEY environment variable not set. "
                "Get your API key from https://console.x.ai/"
            )
        
        # Initialize sync client (async client created in async context)
        self.client = Client(api_key=self.config.api_key)
        
        logger.info(f"GrokBackend initialized with model: {self.config.model}")
    
    def _encode_image(self, image_path: Path) -> str:
        """
        Encode image to base64 for API request.
        
        Args:
            image_path: Path to image file.
        
        Returns:
            Base64 encoded string.
        """
        with open(image_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        return encoded
    
    def _get_mime_type(self, image_path: Path) -> str:
        """
        Get MIME type from image extension.
        
        Args:
            image_path: Path to image file.
        
        Returns:
            MIME type string.
        """
        ext = image_path.suffix.lower()
        mime_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        return mime_types.get(ext, "image/jpeg")
    
    def _record_usage(
        self,
        response,
        truck_id: Optional[str],
        classifier_type: str,
        call_purpose: str
    ) -> None:
        """
        Record API usage for cost tracking.
        
        Args:
            response: API response object with usage attribute.
            truck_id: Truck identifier for tracking.
            classifier_type: Type of classifier (e.g., "body_type").
            call_purpose: Purpose of call (e.g., "reasoning" or "extraction").
        """
        if not ANALYTICS_AVAILABLE:
            return
            
        if not hasattr(response, 'usage') or response.usage is None:
            logger.debug(f"No usage data available for {classifier_type}/{call_purpose}")
            return
        
        usage = response.usage
        cost_tracker = get_cost_tracker()
        
        # Build the API call record
        prompt_tokens = getattr(usage, 'prompt_tokens', 0)
        completion_tokens = getattr(usage, 'completion_tokens', 0)
        reasoning_tokens = getattr(usage, 'reasoning_tokens', 0)
        image_tokens = getattr(usage, 'image_tokens', 0)
        cached_tokens = getattr(usage, 'cached_tokens', 0)
        total_tokens = prompt_tokens + completion_tokens
        
        record = APICallRecord(
            timestamp=datetime.now(),
            truck_id=truck_id,
            classifier_type=classifier_type,
            call_purpose=call_purpose,
            model=self.config.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            image_tokens=image_tokens,
            reasoning_tokens=reasoning_tokens,
            cached_tokens=cached_tokens,
            total_tokens=total_tokens
        )
        
        cost_tracker.record_call(record)
        
        logger.debug(
            f"Recorded usage for {truck_id}/{classifier_type}/{call_purpose}: "
            f"in={prompt_tokens}, out={completion_tokens}, "
            f"reason={reasoning_tokens}, img={image_tokens}"
        )
    
    def classify(
        self,
        image_path: Union[str, Path],
        use_detailed_prompt: Optional[bool] = None,
        truck_id: Optional[str] = None
    ) -> GrokClassificationResult:
        """
        Classify a single image's body type using Grok vision.
        
        Uses a SINGLE-CALL approach with structured decision tree
        to avoid fake confidence from two-chat extraction.
        
        Args:
            image_path: Path to image file.
            use_detailed_prompt: Ignored (kept for API compatibility).
            truck_id: Optional truck ID for cost tracking.
        
        Returns:
            GrokClassificationResult with body type, confidence, and reasoning.
        
        Raises:
            FileNotFoundError: If image file doesn't exist.
            RuntimeError: If API call fails.
        """
        # Validate image path
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        try:
            # Encode image to base64
            base64_image = self._encode_image(image_path)
            mime_type = self._get_mime_type(image_path)
            image_url = f"data:{mime_type};base64,{base64_image}"
            
            # =========================================================
            # SINGLE-CALL STRUCTURED OUTPUT
            # =========================================================
            
            prompt = """Look at this truck image carefully.

This is a full truck with a trailer. Classify the TRAILER type:

- Look at the FRONT of the trailer for a refrigeration unit (boxy equipment) → REEFER
- Enclosed box with NO refrigeration unit on front → DRY_VAN  
- Open flat platform with no walls → FLATBED

Provide your analysis and classification."""
            
            chat = self.client.chat.create(model=self.config.model)
            chat.append(
                user(
                    prompt,
                    image(image_url=image_url, detail=self.config.detail)
                )
            )
            
            # Use structured output - returns (response, parsed_object) tuple
            response, parsed = chat.parse(BodyTypeSchema)
            
            logger.debug(f"Grok structured output: {parsed.body_type} ({parsed.confidence:.2f})")
            logger.debug(f"Reasoning: {parsed.reasoning}")
            
            # Record usage for single API call
            if ANALYTICS_AVAILABLE:
                self._record_usage(response, truck_id, "body_type", "structured_output")
            
            # Convert Pydantic model to GrokClassificationResult
            result = GrokClassificationResult(
                body_type=parsed.body_type.lower(),
                confidence=parsed.confidence,
                reasoning=parsed.reasoning,
                raw_response=response.content,
                model=self.config.model
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Grok classification failed: {e}")
            raise RuntimeError(f"Classification failed: {e}")
    
    async def classify_async(
        self,
        image_path: Union[str, Path],
        use_detailed_prompt: Optional[bool] = None,
        truck_id: Optional[str] = None,
        client: Optional[AsyncClient] = None
    ) -> GrokClassificationResult:
        """
        Async version of classify using AsyncClient.
        
        Classify a single image's body type using Grok vision.
        
        Args:
            image_path: Path to image file.
            use_detailed_prompt: Ignored (kept for API compatibility).
            truck_id: Optional truck ID for cost tracking.
            client: Optional AsyncClient to use (for batch operations).
        
        Returns:
            GrokClassificationResult with body type, confidence, and reasoning.
        
        Raises:
            FileNotFoundError: If image file doesn't exist.
            RuntimeError: If API call fails.
        """
        # Validate image path
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        if client is None:
            raise ValueError("client parameter is required for async classification")
        
        try:
            # Encode image to base64
            base64_image = self._encode_image(image_path)
            mime_type = self._get_mime_type(image_path)
            image_url = f"data:{mime_type};base64,{base64_image}"
            
            # =========================================================
            # SINGLE-CALL STRUCTURED OUTPUT (ASYNC)
            # =========================================================
            
            prompt = """Look at this truck image carefully.

This is a full truck with a trailer. Classify the TRAILER type:

- Look at the FRONT of the trailer for a refrigeration unit (boxy equipment) → REEFER
- Enclosed box with NO refrigeration unit on front → DRY_VAN  
- Open flat platform with no walls → FLATBED

Provide your analysis and classification."""
            
            chat = client.chat.create(model=self.config.model)
            chat.append(
                user(
                    prompt,
                    image(image_url=image_url, detail=self.config.detail)
                )
            )
            
            # Use structured output - returns (response, parsed_object) tuple
            response, parsed = await chat.parse(BodyTypeSchema)
            
            logger.debug(f"Grok structured output: {parsed.body_type} ({parsed.confidence:.2f})")
            logger.debug(f"Reasoning: {parsed.reasoning}")
            
            # Record usage for single API call
            if ANALYTICS_AVAILABLE:
                self._record_usage(response, truck_id, "body_type", "structured_output")
            
            # Convert Pydantic model to GrokClassificationResult
            result = GrokClassificationResult(
                body_type=parsed.body_type.lower(),
                confidence=parsed.confidence,
                reasoning=parsed.reasoning,
                raw_response=response.content,
                model=self.config.model
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Grok classification failed: {e}")
            raise RuntimeError(f"Classification failed: {e}")
    
    async def classify_batch_async(
        self,
        images: List[Union[str, Path]],
        progress_callback: Optional[callable] = None,
        max_concurrent: int = 20
    ) -> List[GrokClassificationResult]:
        """
        Classify multiple images concurrently using asyncio.
        
        Uses asyncio.gather() with semaphore to make concurrent API calls while
        respecting rate limits. With grok-4-1-fast-reasoning's 480 RPM limit,
        we can safely use 10-20 concurrent requests (~6-8 requests/second).
        
        Args:
            images: List of image paths.
            progress_callback: Optional callback(completed_count, total) for progress.
            max_concurrent: Maximum number of concurrent requests (default: 10).
                           Set lower if hitting rate limits (429 errors).
        
        Returns:
            List of GrokClassificationResult, one per image (in same order as input).
        """
        # Create AsyncClient in the current event loop to avoid "different loop" errors
        async_client = AsyncClient(api_key=self.config.api_key)
        
        total = len(images)
        results = [None] * total
        completed = 0
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def classify_with_semaphore(index: int, img: Union[str, Path]):
            """Classify an image with semaphore control."""
            nonlocal completed
            async with semaphore:
                try:
                    result = await self.classify_async(img, truck_id=f"truck_{index}", client=async_client)
                    results[index] = result
                except Exception as e:
                    logger.error(f"Failed to classify {img}: {e}")
                    results[index] = GrokClassificationResult(
                        body_type="unknown",
                        confidence=0.0,
                        reasoning=f"Error: {e}",
                        model=self.config.model
                    )
                
                completed += 1
                if progress_callback:
                    progress_callback(completed, total)
                
                if completed % 10 == 0 or completed == total:
                    progress_pct = (completed / total) * 100
                    print(f"Progress: {progress_pct:.1f}% ({completed}/{total} trucks)")
                    logger.debug(f"Progress: {completed}/{total} ({completed/total*100:.1f}%)")
        
        start_time = time.time()
        
        # Create all tasks and run them concurrently
        tasks = [classify_with_semaphore(i, img) for i, img in enumerate(images)]
        await asyncio.gather(*tasks)
        
        return results
    
    def classify_batch(
        self,
        images: List[Union[str, Path]],
        progress_callback: Optional[callable] = None,
        max_concurrent: int = 20
    ) -> List[GrokClassificationResult]:
        """
        Classify multiple images concurrently.
        
        This is a synchronous wrapper around classify_batch_async().
        Uses asyncio to run concurrent requests efficiently.
        
        Args:
            images: List of image paths.
            progress_callback: Optional callback(completed_count, total) for progress.
            max_concurrent: Maximum number of concurrent requests (default: 10).
        
        Returns:
            List of GrokClassificationResult, one per image (in same order as input).
        """
        return asyncio.run(
            self.classify_batch_async(images, progress_callback, max_concurrent)
        )
    
    @property
    def is_available(self) -> bool:
        """Check if xAI API is available."""
        return XAI_AVAILABLE and bool(self.config.api_key)
    
    def get_model_info(self) -> Dict:
        """Get information about the Grok model."""
        return {
            "backend": "grok",
            "model": self.config.model,
            "temperature": self.config.temperature,
            "provider": "xai",
            "detail": self.config.detail,
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def classify_with_grok(
    image_path: Union[str, Path],
    model: str = "grok-4-fast-non-reasoning",
) -> tuple:
    """
    Quick classification using Grok without managing backend instance.
    
    Args:
        image_path: Path to image file.
        model: Grok model to use.
    
    Returns:
        Tuple of (body_type: str, confidence: float, reasoning: str)
    """
    backend = GrokBackend(GrokConfig(model=model))
    result = backend.classify(image_path)
    return result.body_type, result.confidence, result.reasoning


def check_grok_status() -> Dict[str, Any]:
    """
    Check Grok API status and configuration.
    
    Returns:
        Dictionary with status information.
    """
    if not XAI_AVAILABLE:
        return {
            "available": False,
            "error": "xai-sdk package not installed",
            "api_key_set": False
        }
    
    api_key = os.getenv("XAI_API_KEY")
    
    return {
        "available": XAI_AVAILABLE,
        "api_key_set": bool(api_key),
        "api_key_preview": f"{api_key[:8]}..." if api_key else None,
    }
