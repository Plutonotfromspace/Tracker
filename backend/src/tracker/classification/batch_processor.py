"""
Batch processing script for vehicle classification.

This script processes all truck images in debug_images/ and classifies them
based on their aspect ratio category:

1. FULL TRUCKS (aspect ratio >= 2.0):
   - Body type: REEFER, DRY_VAN, FLATBED
   - Axle type: SPREAD, STANDARD

2. SMALL VEHICLES (aspect ratio < 2.0):
   - Vehicle type: BOBTAIL, VAN, PICKUP, BOX_TRUCK, OTHER

Results are saved to each truck metadata.json file under the
classification section.

Usage:
    python -m src.tracker.classification.batch_processor
    python -m src.tracker.classification.batch_processor --dry-run
    python -m src.tracker.classification.batch_processor --limit 10
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.tracker.classification.body_type_classifier import (
    BodyType,
    BodyTypeClassifier,
    ClassificationResult,
)
from src.tracker.classification.axle_classifier import (
    AxleType,
    AxleClassifier,
    AxleClassificationResult,
)
from src.tracker.classification.small_vehicle_classifier import (
    SmallVehicleType,
    SmallVehicleClassifier,
    SmallVehicleClassificationResult,
)
from src.tracker.data.metadata_writer import MetadataWriter, METADATA_FILENAME
from src.tracker.data.metadata_schema import (
    BodyType as SchemaBodyType,
    AxleType as SchemaAxleType,
    SmallVehicleType as SchemaSmallVehicleType,
)
from src.tracker.data.database import get_session_context
from src.tracker.data import crud
from src.tracker.utils.logging_config import get_logger, setup_logging

logger = get_logger(__name__)

# Version string for tracking which classifier was used
CLASSIFIER_VERSION = "grok-4-1-fast-v1.0"
AXLE_CLASSIFIER_VERSION = "grok-4-1-fast-axle-v1.0"
SMALL_VEHICLE_CLASSIFIER_VERSION = "grok-4-1-fast-small-v1.0"


def update_truck_database(
    truck_folder: Path,
    body_type: Optional[str] = None,
    axle_type: Optional[str] = None,
    small_vehicle_type: Optional[str] = None
) -> None:
    """
    Update truck classification in database based on unique_truck_id from metadata.
    
    Args:
        truck_folder: Path to truck folder
        body_type: Body type classification (optional)
        axle_type: Axle type classification (optional)
        small_vehicle_type: Small vehicle type (optional)
    """
    try:
        # Load metadata to get unique_truck_id
        metadata_file = truck_folder / METADATA_FILENAME
        if not metadata_file.exists():
            logger.debug(f"{truck_folder.name}: No metadata file for database update")
            return
        
        import json
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        unique_truck_id = metadata.get("unique_truck_id")
        if not unique_truck_id:
            logger.debug(f"{truck_folder.name}: No unique_truck_id in metadata")
            return
        
        # Update database
        with get_session_context() as session:
            truck = crud.update_truck_classification(
                session=session,
                unique_truck_id=unique_truck_id,
                body_type=body_type,
                axle_type=axle_type,
                small_vehicle_type=small_vehicle_type
            )
            if truck:
                logger.debug(f"{truck_folder.name}: Updated database for {unique_truck_id}")
            else:
                logger.debug(f"{truck_folder.name}: Truck {unique_truck_id} not found in database")
    except Exception as e:
        logger.warning(f"{truck_folder.name}: Failed to update database: {e}")


def find_truck_folders(
    base_dir: Path,
    pattern: str = "Truck*"
) -> List[Path]:
    """
    Find all truck folders in the base directory.
    
    Args:
        base_dir: Base directory to search (e.g., debug_images/)
        pattern: Glob pattern for folder names
    
    Returns:
        Sorted list of truck folder paths
    """
    if not base_dir.exists():
        logger.warning(f"Base directory does not exist: {base_dir}")
        return []
    
    folders = sorted(base_dir.glob(pattern))
    folders = [f for f in folders if f.is_dir()]
    
    logger.info(f"Found {len(folders)} truck folders in {base_dir}")
    return folders


def find_best_crop(truck_folder: Path, use_mask: bool = False) -> Optional[Path]:
    """
    Find the best crop or mask image in a truck folder.
    
    Prefers entry crops/masks over exit ones.
    
    Args:
        truck_folder: Path to truck folder
        use_mask: If True, look for _mask images instead of _crop
    
    Returns:
        Path to best crop/mask image, or None if none found
    """
    # Determine suffix based on mode
    suffix = "_mask" if use_mask else "_crop"
    
    # Look for images in priority order
    patterns = [
        f"entry_*{suffix}.jpg",
        f"entry_*{suffix}.png",
        f"*{suffix}.jpg",
        f"*{suffix}.png",
    ]
    
    for pattern in patterns:
        images = list(truck_folder.glob(pattern))
        if images:
            # Return first match (should typically only be one)
            return sorted(images)[0]
    
    return None


def map_body_type(classifier_type: BodyType) -> SchemaBodyType:
    """Map classifier BodyType to schema BodyType."""
    mapping = {
        BodyType.REEFER: SchemaBodyType.REEFER,
        BodyType.DRY_VAN: SchemaBodyType.DRY_VAN,
        BodyType.FLATBED: SchemaBodyType.FLATBED,
        BodyType.UNKNOWN: SchemaBodyType.UNKNOWN,
    }
    return mapping.get(classifier_type, SchemaBodyType.UNKNOWN)


def map_axle_type(classifier_type: AxleType) -> SchemaAxleType:
    """Map classifier AxleType to schema AxleType."""
    mapping = {
        AxleType.SPREAD: SchemaAxleType.SPREAD,
        AxleType.STANDARD: SchemaAxleType.STANDARD,
        AxleType.UNKNOWN: SchemaAxleType.UNKNOWN,
    }
    return mapping.get(classifier_type, SchemaAxleType.UNKNOWN)


def map_small_vehicle_type(classifier_type: SmallVehicleType) -> SchemaSmallVehicleType:
    """Map classifier SmallVehicleType to schema SmallVehicleType."""
    mapping = {
        SmallVehicleType.BOBTAIL: SchemaSmallVehicleType.BOBTAIL,
        SmallVehicleType.VAN: SchemaSmallVehicleType.VAN,
        SmallVehicleType.PICKUP: SchemaSmallVehicleType.PICKUP,
        SmallVehicleType.BOX_TRUCK: SchemaSmallVehicleType.BOX_TRUCK,
        SmallVehicleType.OTHER: SchemaSmallVehicleType.OTHER,
        SmallVehicleType.UNKNOWN: SchemaSmallVehicleType.UNKNOWN,
    }
    return mapping.get(classifier_type, SchemaSmallVehicleType.UNKNOWN)


def process_single_truck(
    classifier: BodyTypeClassifier,
    writer: MetadataWriter,
    truck_folder: Path,
    dry_run: bool = False,
    reprocess: bool = False,
    use_mask: bool = False
) -> Tuple[str, Optional[ClassificationResult]]:
    """
    Process a single truck folder.
    
    Args:
        classifier: Body type classifier instance
        writer: Metadata writer instance
        truck_folder: Path to truck folder
        dry_run: If True, don't write to metadata
        reprocess: If True, reclassify even if already classified
        use_mask: If True, use mask images instead of crops
    
    Returns:
        Tuple of (status: str, result: ClassificationResult or None)
        Status is one of: "classified", "skipped", "error", "no_crop"
    """
    truck_id = truck_folder.name
    
    # Find crop/mask image
    crop_path = find_best_crop(truck_folder, use_mask=use_mask)
    if crop_path is None:
        logger.warning(f"{truck_id}: No crop image found")
        return "no_crop", None
    
    # Check if already classified (unless reprocess is True)
    if not reprocess and writer.exists(truck_folder):
        metadata = writer.load(truck_folder)
        if (
            metadata and 
            metadata.classification and 
            metadata.classification.body_type != SchemaBodyType.UNKNOWN
        ):
            logger.debug(
                f"{truck_id}: Already classified as "
                f"{metadata.classification.body_type.value}"
            )
            return "skipped", None
    
    # Classify
    try:
        result = classifier.classify(crop_path, truck_id=truck_id)
        
        logger.info(
            f"{truck_id}: {result.body_type.value} "
            f"({result.confidence:.1%})"
            + (" [uncertain]" if result.is_uncertain else "")
        )
        
        # Update metadata
        if not dry_run:
            body_type_value = map_body_type(result.body_type).value
            updates = {
                "classification": {
                    "body_type": body_type_value,
                    "body_type_confidence": round(result.confidence, 4),
                    "classifier_version": CLASSIFIER_VERSION,
                    "classified_at": datetime.now().isoformat(),
                }
            }
            
            if writer.exists(truck_folder):
                writer.update(truck_folder, updates)
                # Update database
                update_truck_database(truck_folder, body_type=body_type_value)
            else:
                logger.warning(
                    f"{truck_id}: No metadata.json, skipping update"
                )
        
        return "classified", result
    
    except Exception as e:
        logger.error(f"{truck_id}: Classification failed: {e}")
        return "error", None


def process_single_truck_axle(
    axle_classifier: AxleClassifier,
    writer: MetadataWriter,
    truck_folder: Path,
    dry_run: bool = False,
    reprocess: bool = False,
    use_mask: bool = False
) -> Tuple[str, Optional[AxleClassificationResult]]:
    """
    Process axle classification for a single truck folder.
    
    Only runs on full trucks with trailers.
    
    Args:
        axle_classifier: Axle classifier instance
        writer: Metadata writer instance
        truck_folder: Path to truck folder
        dry_run: If True, don't write to metadata
        reprocess: If True, reclassify even if already classified
        use_mask: If True, use mask images instead of crops
    
    Returns:
        Tuple of (status: str, result: AxleClassificationResult or None)
        Status is one of: "classified", "skipped", "error", "no_crop"
    """
    truck_id = truck_folder.name
    
    # Find crop/mask image
    crop_path = find_best_crop(truck_folder, use_mask=use_mask)
    if crop_path is None:
        return "no_crop", None
    
    # Check if already classified (unless reprocess is True)
    if not reprocess and writer.exists(truck_folder):
        metadata = writer.load(truck_folder)
        if (
            metadata and 
            metadata.classification and 
            metadata.classification.axle_type != SchemaAxleType.UNKNOWN
        ):
            logger.debug(
                f"{truck_id}: Axle already classified as "
                f"{metadata.classification.axle_type.value}"
            )
            return "skipped", None
    
    # Classify axle type
    try:
        result = axle_classifier.classify(crop_path, truck_id=truck_id)
        
        logger.info(
            f"{truck_id}: axle={result.axle_type.value} "
            f"({result.confidence:.1%})"
            + (" [uncertain]" if result.is_uncertain else "")
        )
        
        # Update metadata
        if not dry_run:
            axle_type_value = map_axle_type(result.axle_type).value
            updates = {
                "classification": {
                    "axle_type": axle_type_value,
                    "axle_type_confidence": round(result.confidence, 4),
                }
            }
            
            if writer.exists(truck_folder):
                writer.update(truck_folder, updates)
                # Update database
                update_truck_database(truck_folder, axle_type=axle_type_value)
        
        return "classified", result
    
    except Exception as e:
        logger.error(f"{truck_id}: Axle classification failed: {e}")
        return "error", None


def process_small_vehicle(
    classifier: SmallVehicleClassifier,
    writer: MetadataWriter,
    truck_folder: Path,
    dry_run: bool = False,
    reprocess: bool = False,
    use_mask: bool = False
) -> Tuple[str, Optional[SmallVehicleClassificationResult]]:
    """
    Process small vehicle classification for a single folder.
    
    Args:
        classifier: Small vehicle classifier instance
        writer: Metadata writer instance
        truck_folder: Path to truck folder
        dry_run: If True, don't write to metadata
        reprocess: If True, reclassify even if already classified
        use_mask: If True, use mask images instead of crops
    
    Returns:
        Tuple of (status: str, result: SmallVehicleClassificationResult or None)
        Status is one of: "classified", "skipped", "error", "no_crop"
    """
    truck_id = truck_folder.name
    
    # Find crop/mask image
    crop_path = find_best_crop(truck_folder, use_mask=use_mask)
    if crop_path is None:
        logger.warning(f"{truck_id}: No crop image found")
        return "no_crop", None
    
    # Check if already classified (unless reprocess is True)
    if not reprocess and writer.exists(truck_folder):
        metadata = writer.load(truck_folder)
        if (
            metadata and 
            metadata.classification and 
            metadata.classification.small_vehicle_type is not None
        ):
            logger.debug(
                f"{truck_id}: Already classified as small vehicle "
                f"{metadata.classification.small_vehicle_type.value}"
            )
            return "skipped", None
    
    # Classify
    try:
        result = classifier.classify(crop_path, truck_id=truck_id)
        
        logger.info(
            f"{truck_id}: {result.vehicle_type.value} "
            f"({result.confidence:.1%})"
            + (" [uncertain]" if result.is_uncertain else "")
        )
        
        # Update metadata
        if not dry_run:
            small_vehicle_type_value = map_small_vehicle_type(result.vehicle_type).value
            updates = {
                "classification": {
                    "small_vehicle_type": small_vehicle_type_value,
                    "small_vehicle_confidence": round(result.confidence, 4),
                    "classifier_version": SMALL_VEHICLE_CLASSIFIER_VERSION,
                    "classified_at": datetime.now().isoformat(),
                }
            }
            
            if writer.exists(truck_folder):
                writer.update(truck_folder, updates)
                # Update database
                update_truck_database(truck_folder, small_vehicle_type=small_vehicle_type_value)
            else:
                logger.warning(
                    f"{truck_id}: No metadata.json, skipping update"
                )
        
        return "classified", result
    
    except Exception as e:
        logger.error(f"{truck_id}: Small vehicle classification failed: {e}")
        return "error", None


def get_vehicle_category(writer: MetadataWriter, truck_folder: Path) -> str:
    """
    Get the category (full_truck or small_vehicle) from metadata.
    
    Args:
        writer: Metadata writer instance
        truck_folder: Path to truck folder
    
    Returns:
        Category string: "full_truck", "small_vehicle", or "unknown"
    """
    if not writer.exists(truck_folder):
        return "unknown"
    
    metadata = writer.load(truck_folder)
    if metadata and metadata.crop:
        category = metadata.crop.category
        if category:
            return category.value
    
    return "unknown"


def run_batch_classification(
    base_dir: Path,
    dry_run: bool = False,
    limit: Optional[int] = None,
    confidence_threshold: float = 0.6,
    reprocess: bool = False,
    use_mask: bool = False,
    progress_callback: Optional[callable] = None
) -> Dict:
    """
    Run batch classification on all trucks in a directory.
    
    Routes vehicles based on their aspect ratio category:
    
    1. FULL TRUCKS (category="full_truck", aspect ratio >= 2.0):
       - Body type classification: REEFER, DRY_VAN, FLATBED
       - Axle type classification: SPREAD, STANDARD
    
    2. SMALL VEHICLES (category="small_vehicle", aspect ratio < 2.0):
       - Vehicle type classification: BOBTAIL, VAN, PICKUP, BOX_TRUCK, OTHER
    
    Args:
        base_dir: Base directory containing truck folders
        dry_run: If True, don't write to metadata files
        limit: Maximum number of trucks to process (for testing)
        confidence_threshold: Minimum confidence for certain classification
        reprocess: If True, reclassify already-classified trucks
        use_mask: If True, use mask images instead of crops
    
    Returns:
        Dictionary with processing statistics
    """
    logger.info("=" * 60)
    logger.info("VEHICLE CLASSIFICATION - BATCH PROCESSOR")
    logger.info("=" * 60)
    logger.info(f"Base directory: {base_dir}")
    logger.info(f"Dry run: {dry_run}")
    logger.info(f"Confidence threshold: {confidence_threshold}")
    logger.info(f"Reprocess existing: {reprocess}")
    logger.info(f"Using {'mask' if use_mask else 'crop'} images")
    if limit:
        logger.info(f"Limit: {limit} trucks")
    logger.info("-" * 60)
    
    # Find all truck folders
    folders = find_truck_folders(base_dir)
    if limit:
        folders = folders[:limit]
    
    if not folders:
        logger.warning("No truck folders found!")
        return {"total": 0}
    
    writer = MetadataWriter()
    
    # Separate folders by category
    full_truck_folders = []
    small_vehicle_folders = []
    unknown_category_folders = []
    
    for folder in folders:
        category = get_vehicle_category(writer, folder)
        if category == "full_truck":
            full_truck_folders.append(folder)
        elif category == "small_vehicle":
            small_vehicle_folders.append(folder)
        else:
            unknown_category_folders.append(folder)
    
    # Stats tracking
    stats = {
        "total": len(folders),
        "full_truck_count": len(full_truck_folders),
        "small_vehicle_count": len(small_vehicle_folders),
        "unknown_category_count": len(unknown_category_folders),
        # Full truck stats
        "trailer_classified": 0,
        "trailer_skipped": 0,
        "trailer_errors": 0,
        "reefer_count": 0,
        "dry_van_count": 0,
        "flatbed_count": 0,
        # Axle stats
        "axle_classified": 0,
        "axle_skipped": 0,
        "axle_errors": 0,
        "spread_count": 0,
        "standard_count": 0,
        # Small vehicle stats
        "small_classified": 0,
        "small_skipped": 0,
        "small_errors": 0,
        "bobtail_count": 0,
        "van_count": 0,
        "pickup_count": 0,
        "box_truck_count": 0,
        "other_count": 0,
    }
    
    # =========================================================
    # STAGE 1: Full Truck Classification (Trailers) - BATCH MODE
    # =========================================================
    if full_truck_folders:
        logger.info("")
        logger.info("STAGE 1: Full Truck Classification (trailers)")
        logger.info("-" * 40)
        
        body_classifier = BodyTypeClassifier(confidence_threshold=confidence_threshold)
        
        total_classification_items = len(full_truck_folders) * 2 + len(small_vehicle_folders)  # body + axle + small
        
        # Phase 1: Collect images that need classification
        images_to_classify = []
        for folder in full_truck_folders:
            truck_id = folder.name
            crop_path = find_best_crop(folder, use_mask=use_mask)
            
            if crop_path is None:
                logger.warning(f"{truck_id}: No crop image found")
                continue
            
            # Check if already classified
            if not reprocess and writer.exists(folder):
                metadata = writer.load(folder)
                if (metadata and metadata.classification and 
                    metadata.classification.body_type != SchemaBodyType.UNKNOWN):
                    logger.debug(f"{truck_id}: Already classified as {metadata.classification.body_type.value}")
                    stats["trailer_skipped"] += 1
                    continue
            
            images_to_classify.append((crop_path, folder, truck_id))
        
        # Phase 2: Batch classify all images
        if images_to_classify:
            image_paths = [img[0] for img in images_to_classify]
            
            # Create progress wrapper
            def batch_progress(completed, total):
                if progress_callback and total_classification_items > 0:
                    stage_progress = (completed / total_classification_items) * 100
                    progress_callback(
                        stage="classifying",
                        progress=min(stage_progress, 99.9),
                        message=f"Classifying body types: {completed}/{len(images_to_classify)}"
                    )
            
            results = body_classifier.classify_batch(image_paths, progress_callback=batch_progress)
            
            # Phase 3: Process results
            for (_, folder, truck_id), result in zip(images_to_classify, results):
                if result and result.body_type != BodyType.UNKNOWN:
                    stats["trailer_classified"] += 1
                    if result.body_type == BodyType.REEFER:
                        stats["reefer_count"] += 1
                    elif result.body_type == BodyType.DRY_VAN:
                        stats["dry_van_count"] += 1
                    elif result.body_type == BodyType.FLATBED:
                        stats["flatbed_count"] += 1
                    
                    # Update metadata
                    if not dry_run:
                        body_type_value = map_body_type(result.body_type).value
                        updates = {
                            "classification": {
                                "body_type": body_type_value,
                                "body_type_confidence": round(result.confidence, 4),
                                "classifier_version": CLASSIFIER_VERSION,
                                "classified_at": datetime.now().isoformat(),
                            }
                        }
                        if writer.exists(folder):
                            writer.update(folder, updates)
                            update_truck_database(folder, body_type=body_type_value)
                else:
                    logger.error(f"{truck_id}: Classification failed")
                    stats["trailer_errors"] += 1
    
    # =========================================================
    # STAGE 2: Axle Type Classification (Full Trucks Only) - BATCH MODE
    # =========================================================
    if full_truck_folders:
        logger.info("")
        logger.info("STAGE 2: Axle Type Classification (full trucks)")
        logger.info("-" * 40)
        
        axle_classifier = AxleClassifier(confidence_threshold=confidence_threshold)
        
        # Phase 1: Collect images that need axle classification
        images_to_classify = []
        for folder in full_truck_folders:
            truck_id = folder.name
            crop_path = find_best_crop(folder, use_mask=use_mask)
            
            if crop_path is None:
                logger.warning(f"{truck_id}: No crop image found")
                continue
            
            # Check if already classified
            if not reprocess and writer.exists(folder):
                metadata = writer.load(folder)
                if (metadata and metadata.classification and 
                    metadata.classification.axle_type != SchemaAxleType.UNKNOWN):
                    logger.debug(f"{truck_id}: Already axle classified as {metadata.classification.axle_type.value}")
                    stats["axle_skipped"] += 1
                    continue
            
            images_to_classify.append((crop_path, folder, truck_id))
        
        # Phase 2: Batch classify all images
        if images_to_classify:
            image_paths = [img[0] for img in images_to_classify]
            
            # Create progress wrapper
            def batch_progress(completed, total):
                if progress_callback and total_classification_items > 0:
                    completed_items = len(full_truck_folders) + completed  # After body types
                    stage_progress = (completed_items / total_classification_items) * 100
                    progress_callback(
                        stage="classifying",
                        progress=min(stage_progress, 99.9),
                        message=f"Classifying axle types: {completed}/{len(images_to_classify)}"
                    )
            
            results = axle_classifier.classify_batch(image_paths, progress_callback=batch_progress)
            
            # Phase 3: Process results
            for (_, folder, truck_id), result in zip(images_to_classify, results):
                if result and result.axle_type != AxleType.UNKNOWN:
                    stats["axle_classified"] += 1
                    if result.axle_type == AxleType.SPREAD:
                        stats["spread_count"] += 1
                    elif result.axle_type == AxleType.STANDARD:
                        stats["standard_count"] += 1
                    
                    # Update metadata
                    if not dry_run:
                        axle_type_value = map_axle_type(result.axle_type).value
                        updates = {
                            "classification": {
                                "axle_type": axle_type_value,
                                "axle_type_confidence": round(result.confidence, 4),
                                "classifier_version": CLASSIFIER_VERSION,
                                "axle_classified_at": datetime.now().isoformat(),
                            }
                        }
                        if writer.exists(folder):
                            writer.update(folder, updates)
                            update_truck_database(folder, axle_type=axle_type_value)
                else:
                    logger.error(f"{truck_id}: Axle classification failed")
                    stats["axle_errors"] += 1
    
    # =========================================================
    # STAGE 3: Small Vehicle Classification - BATCH MODE
    # =========================================================
    if small_vehicle_folders:
        logger.info("")
        logger.info("STAGE 3: Small Vehicle Classification")
        logger.info("-" * 40)
        
        small_classifier = SmallVehicleClassifier(confidence_threshold=confidence_threshold)
        
        # Phase 1: Collect images that need classification
        images_to_classify = []
        for folder in small_vehicle_folders:
            vehicle_id = folder.name
            crop_path = find_best_crop(folder, use_mask=use_mask)
            
            if crop_path is None:
                logger.warning(f"{vehicle_id}: No crop image found")
                continue
            
            # Check if already classified
            if not reprocess and writer.exists(folder):
                metadata = writer.load(folder)
                if (metadata and metadata.classification and 
                    metadata.classification.vehicle_type != SchemaSmallVehicleType.UNKNOWN):
                    logger.debug(f"{vehicle_id}: Already classified as {metadata.classification.vehicle_type.value}")
                    stats["small_skipped"] += 1
                    continue
            
            images_to_classify.append((crop_path, folder, vehicle_id))
        
        # Phase 2: Batch classify all images
        if images_to_classify:
            image_paths = [img[0] for img in images_to_classify]
            
            # Create progress wrapper
            def batch_progress(completed, total):
                if progress_callback and total_classification_items > 0:
                    completed_items = (len(full_truck_folders) * 2) + completed  # After body + axle
                    stage_progress = (completed_items / total_classification_items) * 100
                    progress_callback(
                        stage="classifying",
                        progress=min(stage_progress, 99.9),
                        message=f"Classifying small vehicles: {completed}/{len(images_to_classify)}"
                    )
            
            results = small_classifier.classify_batch(image_paths, progress_callback=batch_progress)
            
            # Phase 3: Process results
            for (_, folder, vehicle_id), result in zip(images_to_classify, results):
                if result and result.vehicle_type != SmallVehicleType.UNKNOWN:
                    stats["small_classified"] += 1
                    if result.vehicle_type == SmallVehicleType.BOBTAIL:
                        stats["bobtail_count"] += 1
                    elif result.vehicle_type == SmallVehicleType.VAN:
                        stats["van_count"] += 1
                    elif result.vehicle_type == SmallVehicleType.PICKUP:
                        stats["pickup_count"] += 1
                    elif result.vehicle_type == SmallVehicleType.BOX_TRUCK:
                        stats["box_truck_count"] += 1
                    elif result.vehicle_type == SmallVehicleType.OTHER:
                        stats["other_count"] += 1
                    
                    # Update metadata
                    if not dry_run:
                        vehicle_type_value = map_small_vehicle_type(result.vehicle_type).value
                        updates = {
                            "classification": {
                                "vehicle_type": vehicle_type_value,
                                "vehicle_type_confidence": round(result.confidence, 4),
                                "classifier_version": CLASSIFIER_VERSION,
                                "classified_at": datetime.now().isoformat(),
                            }
                        }
                        if writer.exists(folder):
                            writer.update(folder, updates)
                            update_truck_database(folder, small_vehicle_type=vehicle_type_value)
                else:
                    logger.error(f"{vehicle_id}: Classification failed")
                    stats["small_errors"] += 1
    
    # Print summary removed
    
    return stats


def main():
    """Main entry point for batch processing."""
    parser = argparse.ArgumentParser(
        description="Batch classify vehicles by category (full trucks: trailer type + axle; small vehicles: vehicle type)"
    )
    parser.add_argument(
        "--base-dir", "-d",
        type=Path,
        default=Path("outputs/debug_images"),
        help="Base directory containing truck folders (default: outputs/debug_images)"
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Process without writing to metadata files"
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=None,
        help="Maximum number of trucks to process"
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.6,
        help="Confidence threshold (default: 0.6)"
    )
    parser.add_argument(
        "--reprocess", "-r",
        action="store_true",
        help="Reclassify already-classified trucks"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--use-mask", "-m",
        action="store_true",
        help="Use mask images instead of crop images for classification"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level="DEBUG" if args.verbose else "INFO")
    
    try:
        stats = run_batch_classification(
            base_dir=args.base_dir,
            dry_run=args.dry_run,
            limit=args.limit,
            confidence_threshold=args.threshold,
            reprocess=args.reprocess,
            use_mask=args.use_mask
        )
        
        # Exit code based on errors
        if stats.get("errors", 0) > 0:
            sys.exit(1)
        sys.exit(0)
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
