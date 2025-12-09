"""
Small vehicle classifier for non-trailer vehicles.

This module provides a high-level interface for classifying small
aspect ratio vehicles (< 2.0) using Grok VLM (xAI cloud API).

These are vehicles that don't have a full trailer visible:
- BOBTAIL: Semi truck cab without trailer attached
- VAN: Cargo van or delivery van
- PICKUP: Pickup truck
- BOX_TRUCK: Small box truck (non-articulated)
- OTHER: Other small vehicle types

Uses Grok-4 vision model for accurate visual reasoning.
"""

import asyncio
import base64
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

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

class SmallVehicleSchema(BaseModel):
    """Pydantic schema for small vehicle classification structured output."""
    vehicle_type: str = Field(
        description="Vehicle type: 'bobtail', 'van', 'pickup', 'box_truck', or 'other'"
    )
    confidence: float = Field(
        description="Confidence score between 0.0 and 1.0",
        ge=0.0,
        le=1.0
    )
    reasoning: str = Field(
        description="Explanation of the classification decision"
    )


class SmallVehicleType(str, Enum):
    """Enumeration of supported small vehicle types."""
    BOBTAIL = "bobtail"
    VAN = "van"
    PICKUP = "pickup"
    BOX_TRUCK = "box_truck"
    OTHER = "other"
    UNKNOWN = "unknown"


@dataclass
@dataclass
class SmallVehicleClassificationResult:
    """
    Result of small vehicle classification.
    
    Attributes:
        vehicle_type: Predicted vehicle type
        confidence: Confidence score (0-1)
        is_uncertain: True if below confidence threshold
        reasoning: Explanation from the model
        model: Model used for classification
        all_scores: Dictionary of all class scores (for compatibility)
    """
    vehicle_type: SmallVehicleType
    confidence: float
    is_uncertain: bool
    reasoning: str = ""
    model: str = ""
    all_scores: Dict[str, float] = None
    
    def __post_init__(self):
        if self.all_scores is None:
            self.all_scores = {}
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "vehicle_type": self.vehicle_type.value,
            "confidence": round(self.confidence, 4),
            "is_uncertain": self.is_uncertain,
            "reasoning": self.reasoning,
            "all_scores": {k: round(v, 4) for k, v in self.all_scores.items()},
        }
    
    @classmethod
    def uncertain(cls) -> "SmallVehicleClassificationResult":
        """Create an uncertain/unknown result."""
        return cls(
            vehicle_type=SmallVehicleType.UNKNOWN,
            confidence=0.0,
            is_uncertain=True,
            reasoning="Classification failed or uncertain",
            model="",
            all_scores={},
        )


@dataclass
class SmallVehicleConfig:
    """
    Configuration for the small vehicle classifier.
    
    Attributes:
        model: Grok model name (default: grok-4-fast-reasoning)
        api_key: xAI API key (default: from XAI_API_KEY env var)
        temperature: Sampling temperature (0 = deterministic)
        detail: Image detail level ("low", "high", "auto")
        max_retries: Number of retries on failure
    """
    model: str = "grok-4-1-fast"
    api_key: Optional[str] = None
    temperature: float = 0.0
    detail: str = "high"
    max_retries: int = 2
    
    def __post_init__(self):
        """Load API key from environment variable."""
        if self.api_key is None:
            self.api_key = os.getenv("XAI_API_KEY")
            if not self.api_key:
                raise ValueError("XAI_API_KEY environment variable must be set")


class SmallVehicleClassifier:
    """
    Classifier for small aspect ratio vehicles using Grok VLM.
    
    This classifier handles vehicles with aspect ratio < 2.0 that
    don't show a full trailer:
    - BOBTAIL: Semi truck cab without trailer
    - VAN: Cargo van or delivery van
    - PICKUP: Pickup truck
    - BOX_TRUCK: Small box truck (non-articulated)
    - OTHER: Other small vehicles
    
    Example:
        >>> classifier = SmallVehicleClassifier()
        >>> result = classifier.classify("small_vehicle.jpg")
        >>> print(f"{result.vehicle_type.value}: {result.confidence:.0%}")
        >>> print(f"Reasoning: {result.reasoning}")
        bobtail: 95%
        Reasoning: I can see a semi truck cab with no trailer attached...
    """
    
    DEFAULT_CONFIDENCE_THRESHOLD = 0.7
    
    def __init__(
        self,
        config: Optional[SmallVehicleConfig] = None,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ):
        """
        Initialize the small vehicle classifier.
        
        Args:
            config: Configuration options. Uses defaults if None.
            confidence_threshold: Minimum confidence for definite classification.
        """
        if not XAI_AVAILABLE:
            raise ImportError(
                "xai-sdk package not installed. "
                "Install with: pip install xai-sdk"
            )
        
        self.config = config or SmallVehicleConfig()
        self.confidence_threshold = confidence_threshold
        
        if not self.config.api_key:
            raise ValueError(
                "XAI_API_KEY environment variable not set. "
                "Get your API key from https://console.x.ai/"
            )
        
        # Only create sync client (async client created in async context)
        self.client = Client(api_key=self.config.api_key)
        self._initialized = True
        
        logger.info(
            f"SmallVehicleClassifier created with Grok backend "
            f"(threshold={confidence_threshold}, model={self.config.model})"
        )
    
    def _encode_image(self, image_path: Path) -> str:
        """Encode image to base64 for API request."""
        with open(image_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        return encoded
    
    def _get_mime_type(self, image_path: Path) -> str:
        """Get MIME type from image extension."""
        ext = image_path.suffix.lower()
        mime_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        return mime_types.get(ext, "image/jpeg")
    
    def _resolve_image_path(
        self, 
        image_input: Union[str, Path, Image.Image, np.ndarray]
    ) -> Path:
        """Resolve image to a file path."""
        if isinstance(image_input, (str, Path)):
            return Path(image_input)
        
        elif isinstance(image_input, Image.Image):
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
                image_input.save(f.name, 'JPEG')
                return Path(f.name)
        
        elif isinstance(image_input, np.ndarray):
            import tempfile
            from PIL import Image as PILImage
            
            if len(image_input.shape) == 3 and image_input.shape[2] == 3:
                import cv2
                image_input = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
            
            pil_image = PILImage.fromarray(image_input)
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
                pil_image.save(f.name, 'JPEG')
                return Path(f.name)
        
        else:
            raise TypeError(f"Unsupported image type: {type(image_input)}")
    
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
            logger.debug(f"No usage data available for small_vehicle/{call_purpose}")
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
            classifier_type="small_vehicle",
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
            f"Recorded usage for {truck_id}/small_vehicle/{call_purpose}: "
            f"in={prompt_tokens}, out={completion_tokens}, "
            f"reason={reasoning_tokens}, img={image_tokens}"
        )
    
    def classify(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray],
        truck_id: Optional[str] = None
    ) -> SmallVehicleClassificationResult:
        """
        Classify a single image's vehicle type.
        
        Args:
            image_input: Image to classify. Can be path, PIL Image, or numpy array.
            truck_id: Optional truck ID for cost tracking.
        
        Returns:
            SmallVehicleClassificationResult with predicted type and confidence.
        """
        try:
            image_path = self._resolve_image_path(image_input)
            
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")
            
            # Encode image
            base64_image = self._encode_image(image_path)
            mime_type = self._get_mime_type(image_path)
            image_url = f"data:{mime_type};base64,{base64_image}"
            
            # =========================================================
            # SINGLE-CALL STRUCTURED OUTPUT
            # =========================================================
            
            prompt = """Look at this vehicle image carefully.

This is a small/compact vehicle (not a full semi-truck with trailer). Classify it:

BOBTAIL - A semi-truck cab/tractor without any trailer attached. Look for:
- Large cab with typical semi-truck front end
- NO trailer or cargo area behind the cab
- Just the cab unit alone

VAN - A cargo van or delivery van. Look for:
- Single-unit vehicle (cab and cargo in one body)
- Enclosed cargo area behind driver
- Examples: Sprinter van, Transit van, box van

PICKUP - A pickup truck. Look for:
- Open bed in the back
- Cab with open cargo area (not enclosed)
- May have toolbox or cover on bed

BOX_TRUCK - A small box truck (non-articulated). Look for:
- Single-unit truck with enclosed box cargo area
- NOT a semi-truck - cab and box are one connected unit
- Smaller than semi-trucks, like moving trucks

OTHER - If none of the above categories fit.

Provide your analysis and classification."""
            
            chat = self.client.chat.create(model=self.config.model)
            chat.append(
                user(
                    prompt,
                    image(image_url=image_url, detail=self.config.detail)
                )
            )
            
            # Use structured output - returns (response, parsed_object) tuple
            response, parsed = chat.parse(SmallVehicleSchema)
            
            logger.debug(f"Small vehicle structured output: {parsed.vehicle_type} ({parsed.confidence:.2f})")
            logger.debug(f"Reasoning: {parsed.reasoning}")
            
            # Record usage for single API call
            self._record_usage(response, truck_id, "structured_output")
            
            # Convert Pydantic model to SmallVehicleClassificationResult
            is_uncertain = parsed.confidence < self.confidence_threshold
            result = SmallVehicleClassificationResult(
                vehicle_type=SmallVehicleType(parsed.vehicle_type.lower()),
                confidence=parsed.confidence,
                is_uncertain=is_uncertain,
                reasoning=parsed.reasoning,
                model=self.config.model
            )
            return result
            
        except FileNotFoundError as e:
            logger.warning(f"Image not found: {e}")
            return SmallVehicleClassificationResult.uncertain()
        
        except Exception as e:
            logger.error(f"Classification error: {e}")
            return SmallVehicleClassificationResult.uncertain()
    
    async def classify_async(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray],
        truck_id: Optional[str] = None,
        client: Optional[AsyncClient] = None
    ) -> SmallVehicleClassificationResult:
        """
        Async version of classify for concurrent batch processing.
        
        Args:
            image_input: Image to classify. Can be path, PIL Image, or numpy array.
            truck_id: Optional truck ID for cost tracking.
        
        Returns:
            SmallVehicleClassificationResult with predicted type and confidence.
        """
        try:
            image_path = self._resolve_image_path(image_input)
            
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")
            
            if client is None:
                raise ValueError("client parameter is required for async classification")
            
            # Encode image
            base64_image = self._encode_image(image_path)
            mime_type = self._get_mime_type(image_path)
            image_url = f"data:{mime_type};base64,{base64_image}"
            
            # PROMPT FOR SMALL VEHICLE CLASSIFICATION
            prompt = """Look at this vehicle image carefully.

This is a small/compact vehicle (not a full semi-truck with trailer). Classify it:

BOBTAIL - A semi-truck cab/tractor without any trailer attached. Look for:
- Large cab with typical semi-truck front end
- NO trailer or cargo area behind the cab
- Just the cab unit alone

VAN - A cargo van or delivery van. Look for:
- Single-unit vehicle (cab and cargo in one body)
- Enclosed cargo area behind driver
- Examples: Sprinter van, Transit van, box van

PICKUP - A pickup truck. Look for:
- Open bed in the back
- Cab with open cargo area (not enclosed)
- May have toolbox or cover on bed

BOX_TRUCK - A small box truck (non-articulated). Look for:
- Single-unit truck with enclosed box cargo area
- NOT a semi-truck - cab and box are one connected unit
- Smaller than semi-trucks, like moving trucks

OTHER - If none of the above categories fit.

Provide your analysis and classification."""
            
            chat = client.chat.create(model=self.config.model)
            chat.append(
                user(
                    prompt,
                    image(image_url=image_url, detail=self.config.detail)
                )
            )
            
            # Use structured output - returns (response, parsed_object) tuple
            response, parsed = await chat.parse(SmallVehicleSchema)
            
            logger.debug(f"Small vehicle structured output: {parsed.vehicle_type} ({parsed.confidence:.2f})")
            logger.debug(f"Reasoning: {parsed.reasoning}")
            
            # Record usage for single API call
            self._record_usage(response, truck_id, "structured_output")
            
            # Convert Pydantic model to SmallVehicleClassificationResult
            is_uncertain = parsed.confidence < self.confidence_threshold
            result = SmallVehicleClassificationResult(
                vehicle_type=SmallVehicleType(parsed.vehicle_type.lower()),
                confidence=parsed.confidence,
                is_uncertain=is_uncertain,
                reasoning=parsed.reasoning,
                model=self.config.model
            )
            return result
            
        except FileNotFoundError as e:
            logger.warning(f"Image not found: {e}")
            return SmallVehicleClassificationResult.uncertain()
        
        except Exception as e:
            logger.error(f"Classification error: {e}")
            return SmallVehicleClassificationResult.uncertain()
    
    async def classify_batch_async(
        self,
        images: List[Union[str, Path, Image.Image, np.ndarray]],
        progress_callback: Optional[callable] = None,
        max_concurrent: int = 20
    ) -> List[SmallVehicleClassificationResult]:
        """
        Classify multiple images concurrently using async API calls.
        
        Uses asyncio.gather() with a Semaphore to limit concurrent requests.
        This is significantly faster than sequential processing.
        
        Args:
            images: List of images to classify.
            progress_callback: Optional callback(completed, total) for progress.
            max_concurrent: Maximum concurrent API requests (default 10).
        
        Returns:
            List of SmallVehicleClassificationResult, one per image (in same order).
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
        
        async def classify_with_semaphore(index: int, img):
            """Classify one image with semaphore control."""
            nonlocal completed
            async with semaphore:
                result = await self.classify_async(img, client=async_client)
                results[index] = result
                
                # Update progress
                completed += 1
                if progress_callback:
                    progress_callback(completed, total)
                
                # Log progress periodically
                if completed % 10 == 0 or completed == total:
                    elapsed = time.time() - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    progress_pct = (completed / total) * 100
                    print(f"Progress: {progress_pct:.1f}% ({completed}/{total} vehicles)")
                    logger.debug(
                        f"Progress: {completed}/{total} ({completed/total*100:.1f}%) "
                        f"- {rate:.1f} images/sec"
                    )
        
        # Create tasks for all images
        tasks = [
            classify_with_semaphore(i, img) 
            for i, img in enumerate(images)
        ]
        
        # Run all tasks concurrently
        await asyncio.gather(*tasks)
        
        return results
    
    def classify_batch(
        self,
        images: List[Union[str, Path, Image.Image, np.ndarray]],
        progress_callback: Optional[callable] = None,
        max_concurrent: int = 20
    ) -> List[SmallVehicleClassificationResult]:
        """
        Classify multiple images. Uses async implementation under the hood.
        
        Args:
            images: List of images to classify.
            progress_callback: Optional callback(completed, total) for progress.
            max_concurrent: Maximum concurrent API requests (default 20).
        
        Returns:
            List of SmallVehicleClassificationResult, one per image (in same order as input).
        """
        # asyncio.run() creates and manages its own event loop
        return asyncio.run(
            self.classify_batch_async(images, progress_callback, max_concurrent)
        )
    
    def get_classification_stats(
        self,
        results: List[SmallVehicleClassificationResult]
    ) -> Dict:
        """
        Get statistics from a batch of classification results.
        
        Args:
            results: List of classification results.
        
        Returns:
            Dictionary with classification statistics.
        """
        total = len(results)
        if total == 0:
            return {"total": 0}
        
        bobtail_count = sum(1 for r in results if r.vehicle_type == SmallVehicleType.BOBTAIL)
        van_count = sum(1 for r in results if r.vehicle_type == SmallVehicleType.VAN)
        pickup_count = sum(1 for r in results if r.vehicle_type == SmallVehicleType.PICKUP)
        box_truck_count = sum(1 for r in results if r.vehicle_type == SmallVehicleType.BOX_TRUCK)
        other_count = sum(1 for r in results if r.vehicle_type == SmallVehicleType.OTHER)
        unknown_count = sum(1 for r in results if r.vehicle_type == SmallVehicleType.UNKNOWN)
        uncertain_count = sum(1 for r in results if r.is_uncertain)
        
        confidences = [r.confidence for r in results if r.confidence > 0]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        return {
            "total": total,
            "bobtail_count": bobtail_count,
            "van_count": van_count,
            "pickup_count": pickup_count,
            "box_truck_count": box_truck_count,
            "other_count": other_count,
            "unknown_count": unknown_count,
            "uncertain_count": uncertain_count,
            "bobtail_percentage": bobtail_count / total * 100,
            "van_percentage": van_count / total * 100,
            "pickup_percentage": pickup_count / total * 100,
            "box_truck_percentage": box_truck_count / total * 100,
            "other_percentage": other_count / total * 100,
            "uncertain_percentage": uncertain_count / total * 100,
            "average_confidence": avg_confidence,
        }
    
    @property
    def is_initialized(self) -> bool:
        """Check if classifier is fully initialized."""
        return self._initialized
    
    @property
    def model_info(self) -> Dict:
        """Get information about the underlying model."""
        return {
            "backend": "grok",
            "model": self.config.model,
            "temperature": self.config.temperature,
            "confidence_threshold": self.confidence_threshold,
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def classify_small_vehicle(
    image: Union[str, Path, Image.Image, np.ndarray],
    confidence_threshold: float = 0.7
) -> Tuple[str, float]:
    """
    Quick classification of small vehicle without managing a classifier instance.
    
    Args:
        image: Image to classify.
        confidence_threshold: Minimum confidence for definite classification.
    
    Returns:
        Tuple of (vehicle_type: str, confidence: float)
    """
    classifier = SmallVehicleClassifier(confidence_threshold=confidence_threshold)
    result = classifier.classify(image)
    return result.vehicle_type.value, result.confidence
