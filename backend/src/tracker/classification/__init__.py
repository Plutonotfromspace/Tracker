"""
Truck body type, axle type, and small vehicle classification module.

This module provides classification of truck body types, trailer axle
configurations, and small vehicles using Grok VLM (xAI cloud API) 
with visual reasoning.

Classification is routed by aspect ratio category:

FULL TRUCKS (aspect ratio >= 2.0):
  Body Type Classification (3-way):
  - REEFER: Trailer with visible refrigeration unit
  - DRY_VAN: Trailer with flat front, no equipment
  - FLATBED: Open flat platform trailer

  Axle Type Classification (2-way):
  - SPREAD: Wide/spread axles with significant gap
  - STANDARD: Standard axles positioned close together

SMALL VEHICLES (aspect ratio < 2.0):
  Vehicle Type Classification (5-way):
  - BOBTAIL: Truck cab only, no trailer attached
  - VAN: Cargo van or delivery van
  - PICKUP: Pickup truck
  - BOX_TRUCK: Small box truck (non-articulated)
  - OTHER: Other small vehicle types

Example:
    >>> from src.tracker.classification import BodyTypeClassifier, SmallVehicleClassifier
    >>> 
    >>> # For full trucks with trailers
    >>> body_classifier = BodyTypeClassifier()
    >>> result = body_classifier.classify("debug_images/Truck001/entry_crop.jpg")
    >>> print(f"{result.body_type.value}: {result.confidence:.0%}")
    >>> 
    >>> # For small vehicles (bobtails, vans, etc.)
    >>> small_classifier = SmallVehicleClassifier()
    >>> result = small_classifier.classify("debug_images/Truck002/entry_crop.jpg")
    >>> print(f"{result.vehicle_type.value}: {result.confidence:.0%}")
"""

# High-level classifier interface
from src.tracker.classification.body_type_classifier import (
    BodyTypeClassifier,
    BodyType,
    ClassificationResult,
    classify_body_type,
)

# Axle classifier
from src.tracker.classification.axle_classifier import (
    AxleClassifier,
    AxleType,
    AxleClassificationResult,
    AxleClassifierConfig,
)

# Small vehicle classifier
from src.tracker.classification.small_vehicle_classifier import (
    SmallVehicleClassifier,
    SmallVehicleType,
    SmallVehicleClassificationResult,
    SmallVehicleConfig,
    classify_small_vehicle,
)

# Grok Backend (xAI cloud API)
from src.tracker.classification.grok_backend import (
    GrokBackend,
    GrokConfig,
    GrokClassificationResult,
    classify_with_grok,
    check_grok_status,
)

__all__ = [
    # Prompts
    "REEFER_PROMPTS",
    "DRY_VAN_PROMPTS",
    "BOBTAIL_PROMPTS",
    "FLATBED_PROMPTS",
    "BodyTypePrompts",
    "get_three_way_prompts",
    "get_four_way_prompts",
    # Body Type Classifier (trailers)
    "BodyTypeClassifier",
    "BodyType",
    "ClassificationResult",
    "classify_body_type",
    # Axle Classifier
    "AxleClassifier",
    "AxleType",
    "AxleClassificationResult",
    "AxleClassifierConfig",
    # Small Vehicle Classifier
    "SmallVehicleClassifier",
    "SmallVehicleType",
    "SmallVehicleClassificationResult",
    "SmallVehicleConfig",
    "classify_small_vehicle",
    # Grok Backend
    "GrokBackend",
    "GrokConfig",
    "GrokClassificationResult",
    "classify_with_grok",
    "check_grok_status",
]
