"""
Body type classifier for truck trailers.

This module provides a high-level interface for classifying truck
TRAILER types using Grok VLM (xAI cloud API).

Supports 3-WAY classification for full trucks (aspect ratio >= 2.0):
- REEFER: Trailer with visible refrigeration unit on front
- DRY_VAN: Trailer with flat front, no equipment  
- FLATBED: Open flat platform trailer, no enclosed walls

Note: Small vehicles (aspect ratio < 2.0) should use SmallVehicleClassifier
for BOBTAIL, VAN, PICKUP, BOX_TRUCK classification.

Uses Grok-4 vision model for accurate visual reasoning with 94%+ accuracy.
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

from src.tracker.classification.grok_backend import (
    GrokBackend,
    GrokConfig,
    GrokClassificationResult,
)
from src.tracker.utils.logging_config import get_logger

logger = get_logger(__name__)


class BodyType(str, Enum):
    """Enumeration of supported trailer body types."""
    REEFER = "reefer"
    DRY_VAN = "dry_van"
    FLATBED = "flatbed"
    UNKNOWN = "unknown"


@dataclass
class ClassificationResult:
    """
    Result of body type classification.
    
    Attributes:
        body_type: Predicted body type
        confidence: Confidence score (0-1)
        is_uncertain: True if below confidence threshold
        reasoning: Explanation from the model
        all_scores: Dictionary of all class scores (for compatibility)
    """
    body_type: BodyType
    confidence: float
    is_uncertain: bool
    reasoning: str = ""
    all_scores: Dict[str, float] = None
    
    def __post_init__(self):
        if self.all_scores is None:
            self.all_scores = {}
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "body_type": self.body_type.value,
            "confidence": round(self.confidence, 4),
            "is_uncertain": self.is_uncertain,
            "reasoning": self.reasoning,
            "all_scores": {k: round(v, 4) for k, v in self.all_scores.items()},
        }
    
    @classmethod
    def uncertain(cls) -> "ClassificationResult":
        """Create an uncertain/unknown result."""
        return cls(
            body_type=BodyType.UNKNOWN,
            confidence=0.0,
            is_uncertain=True,
            reasoning="Classification failed or uncertain",
            all_scores={},
        )


class BodyTypeClassifier:
    """
    High-level classifier for truck trailer types using Grok VLM.
    
    This classifier uses xAI's Grok-4 vision model with reasoning
    capabilities to accurately classify trailer types:
    - REEFER: Has visible refrigeration unit on trailer front
    - DRY_VAN: Has flat front with no equipment  
    - FLATBED: Open flat trailer platform
    
    Note: For small vehicles (aspect ratio < 2.0), use SmallVehicleClassifier.
    
    Features:
        - 3-way trailer classification
        - Visual reasoning with explanations
        - 94%+ accuracy on test dataset
        - Cloud API (no local GPU required)
    
    Example:
        >>> classifier = BodyTypeClassifier()
        >>> result = classifier.classify("truck_image.jpg")
        >>> print(f"{result.body_type.value}: {result.confidence:.0%}")
        >>> print(f"Reasoning: {result.reasoning}")
        reefer: 100%
        Reasoning: I can see a refrigeration unit on the front of the trailer...
    """
    
    # Default confidence threshold - below this, result is marked uncertain
    DEFAULT_CONFIDENCE_THRESHOLD = 0.7
    
    def __init__(
        self,
        config: Optional[GrokConfig] = None,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        **kwargs  # Accept but ignore legacy args like use_ensemble, include_bobtail
    ):
        """
        Initialize the body type classifier.
        
        Args:
            config: Grok backend configuration. Uses defaults if None.
            confidence_threshold: Minimum confidence for definite classification.
                                  Results below this are marked as uncertain.
        """
        self.confidence_threshold = confidence_threshold
        
        # Initialize Grok backend
        self._backend = GrokBackend(config)
        self._initialized = True
        
        logger.info(
            f"BodyTypeClassifier created with Grok backend "
            f"(threshold={confidence_threshold}, model={self._backend.config.model})"
        )
    
    def classify(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        truck_id: Optional[str] = None
    ) -> ClassificationResult:
        """
        Classify a single image's body type.
        
        Args:
            image: Image to classify. Can be:
                   - Path to image file (str or Path)
                   - PIL Image
                   - NumPy array (will be saved temporarily)
            truck_id: Optional truck ID for cost tracking.
        
        Returns:
            ClassificationResult with predicted body type and confidence.
        
        Example:
            >>> result = classifier.classify("truck.jpg")
            >>> if result.is_uncertain:
            ...     print("Low confidence, manual review needed")
            >>> else:
            ...     print(f"Body type: {result.body_type.value}")
        """
        try:
            # Handle different image types
            image_path = self._resolve_image_path(image)
            
            # Classify with Grok
            grok_result = self._backend.classify(image_path, truck_id=truck_id)
            return self._convert_result(grok_result)
        
        except FileNotFoundError as e:
            logger.warning(f"Image not found: {e}")
            return ClassificationResult.uncertain()
        
        except Exception as e:
            logger.error(f"Classification error: {e}")
            return ClassificationResult.uncertain()
    
    def classify_batch(
        self,
        images: List[Union[str, Path, Image.Image, np.ndarray]],
        progress_callback: Optional[callable] = None
    ) -> List[ClassificationResult]:
        """
        Classify multiple images concurrently using async API calls.
        
        This method uses asyncio.run() to execute concurrent API calls,
        significantly improving throughput for batch classification.
        
        Args:
            images: List of images to classify.
            progress_callback: Optional callback(index, total) for progress.
        
        Returns:
            List of ClassificationResult, one per image.
        """
        import asyncio
        
        # Use asyncio.run() which creates and manages its own event loop
        return asyncio.run(self.classify_batch_async(images, progress_callback))
    
    async def classify_batch_async(
        self,
        images: List[Union[str, Path, Image.Image, np.ndarray]],
        progress_callback: Optional[callable] = None
    ) -> List[ClassificationResult]:
        """
        Async version of classify_batch for concurrent API calls.
        
        Args:
            images: List of images to classify.
            progress_callback: Optional callback(index, total) for progress.
        
        Returns:
            List of ClassificationResult, one per image.
        """
        # Resolve all image paths first
        image_paths = [self._resolve_image_path(img) for img in images]
        
        # Call backend's async batch method
        grok_results = await self._backend.classify_batch_async(
            image_paths,
            progress_callback=progress_callback
        )
        
        # Convert results
        return [self._convert_result(gr) for gr in grok_results]
    
    def _resolve_image_path(
        self, 
        image: Union[str, Path, Image.Image, np.ndarray]
    ) -> Path:
        """
        Resolve image to a file path.
        
        If image is a PIL Image or numpy array, it's saved to a temp file.
        """
        if isinstance(image, (str, Path)):
            return Path(image)
        
        elif isinstance(image, Image.Image):
            # Save PIL image to temp file
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
                image.save(f.name, 'JPEG')
                return Path(f.name)
        
        elif isinstance(image, np.ndarray):
            # Convert numpy array to PIL and save
            import tempfile
            from PIL import Image as PILImage
            
            # Assume BGR if 3 channels (OpenCV format)
            if len(image.shape) == 3 and image.shape[2] == 3:
                import cv2
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            pil_image = PILImage.fromarray(image)
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
                pil_image.save(f.name, 'JPEG')
                return Path(f.name)
        
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")
    
    def _convert_result(
        self,
        grok_result: GrokClassificationResult
    ) -> ClassificationResult:
        """Convert Grok result to ClassificationResult."""
        # Map class names to BodyType enum
        predicted = grok_result.body_type.lower()
        confidence = grok_result.confidence
        
        # Determine body type
        body_type_map = {
            "reefer": BodyType.REEFER,
            "dry_van": BodyType.DRY_VAN,
            "flatbed": BodyType.FLATBED,
        }
        body_type = body_type_map.get(predicted, BodyType.UNKNOWN)
        
        # Check if uncertain
        is_uncertain = confidence < self.confidence_threshold
        
        if is_uncertain:
            logger.debug(
                f"Low confidence classification: {predicted} ({confidence:.0%})"
            )
        
        # Build scores dict (single class for Grok)
        all_scores = {predicted: confidence}
        
        return ClassificationResult(
            body_type=body_type,
            confidence=confidence,
            is_uncertain=is_uncertain,
            reasoning=grok_result.reasoning,
            all_scores=all_scores,
        )
    
    def get_classification_stats(
        self,
        results: List[ClassificationResult]
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
        
        reefer_count = sum(1 for r in results if r.body_type == BodyType.REEFER)
        dry_van_count = sum(1 for r in results if r.body_type == BodyType.DRY_VAN)
        flatbed_count = sum(1 for r in results if r.body_type == BodyType.FLATBED)
        unknown_count = sum(1 for r in results if r.body_type == BodyType.UNKNOWN)
        uncertain_count = sum(1 for r in results if r.is_uncertain)
        
        confidences = [r.confidence for r in results if r.confidence > 0]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        return {
            "total": total,
            "reefer_count": reefer_count,
            "dry_van_count": dry_van_count,
            "flatbed_count": flatbed_count,
            "unknown_count": unknown_count,
            "uncertain_count": uncertain_count,
            "reefer_percentage": reefer_count / total * 100,
            "dry_van_percentage": dry_van_count / total * 100,
            "flatbed_percentage": flatbed_count / total * 100,
            "uncertain_percentage": uncertain_count / total * 100,
            "average_confidence": avg_confidence,
        }
    
    @property
    def is_initialized(self) -> bool:
        """Check if classifier is fully initialized."""
        return self._initialized and self._backend.is_available
    
    @property
    def model_info(self) -> Dict:
        """Get information about the underlying model."""
        return {
            **self._backend.get_model_info(),
            "confidence_threshold": self.confidence_threshold,
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def classify_body_type(
    image: Union[str, Path, Image.Image, np.ndarray],
    confidence_threshold: float = 0.7
) -> Tuple[str, float]:
    """
    Quick classification of body type without managing a classifier instance.
    
    Note: This creates a new classifier each time. For batch processing,
    create a BodyTypeClassifier instance and reuse it.
    
    Args:
        image: Image to classify.
        confidence_threshold: Minimum confidence for definite classification.
    
    Returns:
        Tuple of (body_type: str, confidence: float)
    
    Example:
        >>> body_type, conf = classify_body_type("truck.jpg")
        >>> print(f"{body_type}: {conf:.0%}")
    """
    classifier = BodyTypeClassifier(confidence_threshold=confidence_threshold)
    result = classifier.classify(image)
    return result.body_type.value, result.confidence
