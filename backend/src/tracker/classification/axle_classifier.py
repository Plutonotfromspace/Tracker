"""
Axle type classifier for trailer rear axle configuration.

Classifies trailers into two axle types:
- SPREAD: Wide/spread axles with significant gap (~10ft spacing)
- STANDARD: Standard axles positioned close together (~4ft spacing)

Uses the same two-chat Grok approach as body_type_classifier for consistency
and proven 94% accuracy pattern.

Only runs on trailers (non-bobtail trucks).
"""

import asyncio
import base64
import os
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Callable, List, Optional, Union

from pydantic import BaseModel, Field

from src.tracker.utils.logging_config import get_logger

logger = get_logger(__name__)

# Try to import analytics (may not be available during initial setup)
try:
    from src.tracker.analytics import get_cost_tracker
    from src.tracker.analytics.cost_tracker import APICallRecord
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
# PYDANTIC SCHEMA FOR STRUCTURED OUTPUT
# =============================================================================

class AxleTypeSchema(BaseModel):
    """Pydantic schema for axle type classification structured output."""
    axle_type: str = Field(
        description="Axle type classification: 'spread' or 'standard'"
    )
    confidence: float = Field(
        description="Confidence score between 0.0 and 1.0",
        ge=0.0,
        le=1.0
    )
    reasoning: str = Field(
        description="Explanation of the classification decision"
    )


class AxleType(str, Enum):
    """
    Trailer axle configuration types.
    
    For this classifier, we focus on the two main types:
    - SPREAD: Non-standard center-to-center distance (wide gap)
    - STANDARD: Standard center-to-center distance (close together)
    """
    SPREAD = "spread"      # Wide/spread axles - significant gap between them
    STANDARD = "standard"  # Standard axles - close together
    UNKNOWN = "unknown"    # Could not determine


@dataclass
@dataclass
class AxleClassificationResult:
    """
    Result from axle type classification.
    
    Attributes:
        axle_type: Classified axle type (SPREAD, STANDARD, UNKNOWN)
        confidence: Confidence score (0.0 to 1.0)
        reasoning: Model's reasoning for the classification
        raw_response: Raw model response
    """
    axle_type: AxleType
    confidence: float
    reasoning: str = ""
    raw_response: str = ""
    model: str = ""
    
    @property
    def is_uncertain(self) -> bool:
        """Whether classification confidence is below threshold."""
        return self.confidence < 0.6 or self.axle_type == AxleType.UNKNOWN


@dataclass
class AxleClassifierConfig:
    """
    Configuration for the axle classifier.
    """
    model: str = "grok-4-1-fast"
    api_key: Optional[str] = None
    temperature: float = 0.0  # Deterministic
    detail: str = "high"  # High detail for axle spacing detection
    max_retries: int = 2
    
    def __post_init__(self):
        """Load API key from environment variable."""
        if self.api_key is None:
            self.api_key = os.getenv("XAI_API_KEY")
            if not self.api_key:
                raise ValueError("XAI_API_KEY environment variable must be set")


class AxleClassifier:
    """
    Classifies trailer axle configuration using Grok vision.
    
    Uses the same two-chat approach as body_type_classifier:
    1. Chat 1: Ask model to analyze rear axle spacing
    2. Chat 2: Extract structured JSON classification
    
    Only should be run on trailers (non-bobtail trucks).
    """
    
    def __init__(
        self,
        config: Optional[AxleClassifierConfig] = None,
        confidence_threshold: float = 0.6
    ):
        """
        Initialize the axle classifier.
        
        Args:
            config: Classifier configuration (uses defaults if None)
            confidence_threshold: Minimum confidence for certain classification
        """
        if not XAI_AVAILABLE:
            raise ImportError(
                "xai-sdk package not installed. Install with: pip install xai-sdk"
            )
        
        self.config = config or AxleClassifierConfig()
        self.confidence_threshold = confidence_threshold
        
        if not self.config.api_key:
            raise ValueError(
                "XAI_API_KEY environment variable not set. "
                "Get your API key from https://console.x.ai/"
            )
        
        # Only create sync client (async client created in async context)
        self.client = Client(api_key=self.config.api_key)
        
        logger.info(
            f"AxleClassifier initialized with model: {self.config.model}"
        )
    
    def _encode_image(self, image_path: Path) -> str:
        """Encode image to base64."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    
    def _get_mime_type(self, image_path: Path) -> str:
        """Get MIME type from file extension."""
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
        call_purpose: str
    ) -> None:
        """
        Record API usage for cost tracking.
        
        Args:
            response: API response object with usage attribute.
            truck_id: Truck identifier for tracking.
            call_purpose: Purpose of call (e.g., "reasoning" or "extraction").
        """
        if not ANALYTICS_AVAILABLE:
            return
            
        if not hasattr(response, 'usage') or response.usage is None:
            logger.debug(f"No usage data available for axle_type/{call_purpose}")
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
            classifier_type="axle_type",
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
            f"Recorded usage for {truck_id}/axle_type/{call_purpose}: "
            f"in={prompt_tokens}, out={completion_tokens}, "
            f"reason={reasoning_tokens}, img={image_tokens}"
        )
    
    def classify(
        self,
        image_path: Union[str, Path],
        truck_id: Optional[str] = None
    ) -> AxleClassificationResult:
        """
        Classify axle type from a truck image.
        
        Uses two-chat approach:
        1. Chat 1: Analyze the rear axle spacing
        2. Chat 2: Extract JSON classification
        
        Args:
            image_path: Path to truck/trailer image
            truck_id: Optional truck ID for cost tracking
            
        Returns:
            AxleClassificationResult with axle_type and confidence
            
        Raises:
            FileNotFoundError: If image doesn't exist
            RuntimeError: If API call fails
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        try:
            # Encode image
            base64_image = self._encode_image(image_path)
            mime_type = self._get_mime_type(image_path)
            image_url = f"data:{mime_type};base64,{base64_image}"
            
            # =========================================================
            # SINGLE-CALL STRUCTURED OUTPUT
            # =========================================================
            
            prompt = """Look at this truck/trailer image carefully.

Focus on the REAR AXLE GROUP of the trailer - this is the group of wheels at the very back end of the trailer (not the cab wheels, not the front trailer wheels).

The rear axle group typically has TWO axle sets (4 wheels visible from the side - 2 pairs).

Measure the SPACING BETWEEN THE TWO AXLES within this rear group:

- SPREAD: The two rear axles have a WIDE GAP between them - you can clearly see road/ground between the two axle sets. The spacing looks noticeably wider than normal.

- STANDARD: The two rear axles are positioned CLOSE TOGETHER with minimal gap - they almost touch or have very little space between them.

Provide your analysis and classification of the axle configuration."""
            
            chat = self.client.chat.create(model=self.config.model)
            chat.append(
                user(
                    prompt,
                    image(image_url=image_url, detail=self.config.detail)
                )
            )
            
            # Use structured output - returns (response, parsed_object) tuple
            response, parsed = chat.parse(AxleTypeSchema)
            
            logger.debug(f"Axle structured output: {parsed.axle_type} ({parsed.confidence:.2f})")
            logger.debug(f"Reasoning: {parsed.reasoning}")
            
            # Record usage for single API call
            self._record_usage(response, truck_id, "structured_output")
            
            # Convert Pydantic model to AxleClassificationResult
            result = AxleClassificationResult(
                axle_type=AxleType(parsed.axle_type.lower()),
                confidence=parsed.confidence,
                reasoning=parsed.reasoning,
                model=self.config.model
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Axle classification failed: {e}")
            raise RuntimeError(f"Axle classification failed: {e}")
    
    async def classify_async(
        self,
        image_path: Union[str, Path],
        truck_id: Optional[str] = None,
        client: Optional[AsyncClient] = None
    ) -> AxleClassificationResult:
        """
        Async version of classify for concurrent batch processing.
        
        Uses two-chat approach:
        1. Chat 1: Analyze the rear axle spacing
        2. Chat 2: Extract JSON classification
        
        Args:
            image_path: Path to truck/trailer image
            truck_id: Optional truck ID for cost tracking
            
        Returns:
            AxleClassificationResult with axle_type and confidence
            
        Raises:
            FileNotFoundError: If image doesn't exist
            RuntimeError: If API call fails
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        if client is None:
            raise ValueError("client parameter is required for async classification")
        
        try:
            # Encode image
            base64_image = self._encode_image(image_path)
            mime_type = self._get_mime_type(image_path)
            image_url = f"data:{mime_type};base64,{base64_image}"
            
            # SINGLE-CALL STRUCTURED OUTPUT (ASYNC)
            prompt = """Look at this truck/trailer image carefully.

Focus on the REAR AXLE GROUP of the trailer - this is the group of wheels at the very back end of the trailer (not the cab wheels, not the front trailer wheels).

The rear axle group typically has TWO axle sets (4 wheels visible from the side - 2 pairs).

Measure the SPACING BETWEEN THE TWO AXLES within this rear group:

- SPREAD: The two rear axles have a WIDE GAP between them - you can clearly see road/ground between the axles sets. The spacing looks noticeably wider than normal.

- STANDARD: The two rear axles are positioned CLOSE TOGETHER with minimal gap - they almost touch or have very little space between them.

Provide your analysis and classification of the axle configuration."""
            
            chat = client.chat.create(model=self.config.model)
            chat.append(
                user(
                    prompt,
                    image(image_url=image_url, detail=self.config.detail)
                )
            )
            
            # Use structured output - returns (response, parsed_object) tuple
            response, parsed = await chat.parse(AxleTypeSchema)
            
            logger.debug(f"Axle structured output: {parsed.axle_type} ({parsed.confidence:.2f})")
            logger.debug(f"Reasoning: {parsed.reasoning}")
            
            # Record usage for single API call
            self._record_usage(response, truck_id, "structured_output")
            
            # Convert Pydantic model to AxleClassificationResult
            result = AxleClassificationResult(
                axle_type=AxleType(parsed.axle_type.lower()),
                confidence=parsed.confidence,
                reasoning=parsed.reasoning,
                model=self.config.model
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Axle classification failed: {e}")
            raise RuntimeError(f"Axle classification failed: {e}")
    
    async def classify_batch_async(
        self,
        images: List[Union[str, Path]],
        progress_callback: Optional[callable] = None,
        max_concurrent: int = 20
    ) -> List[AxleClassificationResult]:
        """
        Classify multiple images concurrently using async API calls.
        
        Uses asyncio.gather() with a Semaphore to limit concurrent requests.
        This is significantly faster than sequential processing.
        
        Args:
            images: List of image paths
            progress_callback: Optional callback(completed, total) for progress
            max_concurrent: Maximum concurrent API requests (default 10)
            
        Returns:
            List of classification results (in same order as input)
        """
        # Create AsyncClient in the current event loop to avoid "different loop" errors
        async_client = AsyncClient(api_key=self.config.api_key)
        
        # Create AsyncClient in the current event loop to avoid "different loop" errors
        async_client = AsyncClient(api_key=self.config.api_key)
        
        total = len(images)
        results = [None] * total  # Pre-allocate to preserve order
        completed = 0
        start_time = time.time()
        
        # Semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def classify_with_semaphore(index: int, img_path):
            """Classify one image with semaphore control."""
            nonlocal completed
            async with semaphore:
                try:
                    result = await self.classify_async(img_path, client=async_client)
                    results[index] = result
                except Exception as e:
                    logger.error(f"Failed to classify {img_path}: {e}")
                    results[index] = AxleClassificationResult(
                        axle_type=AxleType.UNKNOWN,
                        confidence=0.0,
                        reasoning=f"Error: {e}"
                    )
                
                # Update progress
                completed += 1
                if progress_callback:
                    progress_callback(completed, total)
                
                # Log progress periodically
                if completed % 10 == 0 or completed == total:
                    elapsed = time.time() - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    progress_pct = (completed / total) * 100
                    print(f"Progress: {progress_pct:.1f}% ({completed}/{total} trucks)")
                    logger.debug(
                        f"Progress: {completed}/{total} ({completed/total*100:.1f}%) "
                        f"- {rate:.1f} images/sec"
                    )
        
        # Create tasks for all images
        tasks = [
            classify_with_semaphore(i, img_path) 
            for i, img_path in enumerate(images)
        ]
        
        # Run all tasks concurrently
        await asyncio.gather(*tasks)
        
        return results
    
    def classify_batch(
        self,
        images: List[Union[str, Path]],
        progress_callback: Optional[Callable[[int, int], None]] = None,
        max_concurrent: int = 20
    ) -> List[AxleClassificationResult]:
        """
        Classify multiple images. Uses async implementation under the hood.
        
        Args:
            images: List of image paths
            progress_callback: Optional callback(completed, total) for progress
            max_concurrent: Maximum concurrent API requests (default 10)
            
        Returns:
            List of classification results (in same order as input)
        """
        # asyncio.run() creates and manages its own event loop
        return asyncio.run(
            self.classify_batch_async(images, progress_callback, max_concurrent)
        )
    
    def get_classification_stats(
        self,
        results: List[AxleClassificationResult]
    ) -> dict:
        """
        Calculate statistics from classification results.
        
        Args:
            results: List of classification results
            
        Returns:
            Dictionary with stats
        """
        if not results:
            return {}
        
        spread_count = sum(1 for r in results if r.axle_type == AxleType.SPREAD)
        standard_count = sum(1 for r in results if r.axle_type == AxleType.STANDARD)
        unknown_count = sum(1 for r in results if r.axle_type == AxleType.UNKNOWN)
        uncertain_count = sum(1 for r in results if r.is_uncertain)
        
        confidences = [r.confidence for r in results if r.confidence > 0]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        return {
            "spread_count": spread_count,
            "standard_count": standard_count,
            "unknown_count": unknown_count,
            "uncertain_count": uncertain_count,
            "average_confidence": avg_confidence,
            "total": len(results)
        }
