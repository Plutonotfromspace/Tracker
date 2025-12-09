"""
Compile cropped truck entry images into canvas images.

Stacks images side by side, top to bottom until canvas is full,
then creates a new canvas.

Usage:
    python -m tools.compile_crops [--input DIR] [--output DIR] [--size SIZE]
"""

import argparse
import glob
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np


@dataclass
class ImageData:
    """Data for a loaded truck image."""
    
    image: np.ndarray
    path: str
    truck: str
    height: int
    width: int
    resized: Optional[np.ndarray] = None
    resized_width: int = 0


def compile_crops(
    input_dir: str = "outputs/debug_images",
    output_dir: str = "outputs/compiled_canvases",
    canvas_size: int = 2000,
    padding: int = 5
) -> List[str]:
    """
    Compile all crop images from truck folders into canvas images.
    
    Args:
        input_dir: Directory containing Truck### folders.
        output_dir: Directory to save compiled canvases.
        canvas_size: Size of square canvas in pixels.
        padding: Pixels between images.
    
    Returns:
        List of paths to created canvas images.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all truck folders
    truck_folders = sorted(glob.glob(os.path.join(input_dir, "Truck*")))
    print(f"Found {len(truck_folders)} truck folders")
    
    # Find all crop images
    crop_files = []
    for folder in truck_folders:
        # Look for the entry crop image
        crops = glob.glob(os.path.join(folder, "entry_*_crop.jpg"))
        if crops:
            crop_files.append(crops[0])
    
    print(f"Found {len(crop_files)} crop images")
    
    if not crop_files:
        print("No crop images found!")
        return []
    
    # Load all images
    images = _load_images(crop_files)
    print(f"Loaded {len(images)} images")
    
    # Calculate uniform height
    target_height = _calculate_target_height(images, canvas_size)
    print(f"Target row height: {target_height}px")
    
    # Resize all images
    _resize_images(images, target_height)
    
    # Pack images into canvases
    canvases = _pack_into_canvases(images, output_dir, canvas_size, padding)
    
    # Print summary
    _print_summary(images, canvases, output_dir)
    
    return canvases


def _load_images(crop_files: List[str]) -> List[ImageData]:
    """Load images from file paths."""
    images = []
    for crop_file in crop_files:
        img = cv2.imread(crop_file)
        if img is not None:
            truck_num = os.path.basename(os.path.dirname(crop_file))
            images.append(ImageData(
                image=img,
                path=crop_file,
                truck=truck_num,
                height=img.shape[0],
                width=img.shape[1]
            ))
    return images


def _calculate_target_height(images: List[ImageData], canvas_size: int) -> int:
    """Calculate optimal row height based on image sizes."""
    heights = [img.height for img in images]
    target_height = int(np.median(heights))
    
    # Cap target height to fit reasonable number of rows
    max_height = canvas_size // 4  # At least 4 rows
    return min(target_height, max_height)


def _resize_images(images: List[ImageData], target_height: int) -> None:
    """Resize all images to target height maintaining aspect ratio."""
    for img_data in images:
        h, w = img_data.image.shape[:2]
        scale = target_height / h
        new_width = int(w * scale)
        img_data.resized = cv2.resize(img_data.image, (new_width, target_height))
        img_data.resized_width = new_width


def _pack_into_canvases(
    images: List[ImageData],
    output_dir: str,
    canvas_size: int,
    padding: int
) -> List[str]:
    """Pack images into canvas images."""
    canvases = []
    canvas_num = 1
    
    # Current position on canvas
    x = padding
    y = padding
    
    # Calculate row height from first image (ensure resized exists)
    if not images or images[0].resized is None:
        return canvases
    row_height = images[0].resized.shape[0]
    
    # Create first canvas (dark gray background)
    canvas = np.ones((canvas_size, canvas_size, 3), dtype=np.uint8) * 40
    images_on_canvas = 0
    
    for img_data in images:
        resized = img_data.resized
        if resized is None:
            continue
        w = img_data.resized_width
        
        # Check if image fits in current row
        if x + w + padding > canvas_size:
            # Move to next row
            x = padding
            y += row_height + padding
        
        # Check if we need a new canvas
        if y + row_height + padding > canvas_size:
            # Save current canvas if it has images
            if images_on_canvas > 0:
                canvas_path = os.path.join(
                    output_dir, f"canvas_{canvas_num:03d}.jpg"
                )
                cv2.imwrite(canvas_path, canvas)
                print(f"Saved {canvas_path} with {images_on_canvas} images")
                canvases.append(canvas_path)
                canvas_num += 1
            
            # Create new canvas
            canvas = np.ones(
                (canvas_size, canvas_size, 3), dtype=np.uint8
            ) * 40
            x = padding
            y = padding
            images_on_canvas = 0
        
        # Place image on canvas
        canvas[y:y + row_height, x:x + w] = resized
        
        # Add truck label
        label = img_data.truck.replace("Truck", "T")
        cv2.putText(
            canvas, label, (x + 5, y + 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
        )
        
        x += w + padding
        images_on_canvas += 1
    
    # Save final canvas if it has images
    if images_on_canvas > 0:
        canvas_path = os.path.join(output_dir, f"canvas_{canvas_num:03d}.jpg")
        cv2.imwrite(canvas_path, canvas)
        print(f"Saved {canvas_path} with {images_on_canvas} images")
        canvases.append(canvas_path)
    
    return canvases


def _print_summary(
    images: List[ImageData],
    canvases: List[str],
    output_dir: str
) -> None:
    """Print compilation summary."""
    print(f"\n{'=' * 60}")
    print("COMPILATION COMPLETE")
    print(f"{'=' * 60}")
    print(f"Total images compiled: {len(images)}")
    print(f"Total canvases created: {len(canvases)}")
    print(f"Output directory: {output_dir}/")
    print(f"{'=' * 60}")


def main() -> None:
    """Main entry point for the crop compiler."""
    parser = argparse.ArgumentParser(
        description="Compile truck crop images into canvases"
    )
    parser.add_argument(
        "--input", "-i",
        default="outputs/debug_images",
        help="Input directory containing Truck### folders"
    )
    parser.add_argument(
        "--output", "-o",
        default="outputs/compiled_canvases",
        help="Output directory for canvas images"
    )
    parser.add_argument(
        "--size", "-s",
        type=int,
        default=2000,
        help="Canvas size in pixels (default: 2000)"
    )
    parser.add_argument(
        "--padding", "-p",
        type=int,
        default=5,
        help="Padding between images (default: 5)"
    )
    
    args = parser.parse_args()
    
    compile_crops(
        input_dir=args.input,
        output_dir=args.output,
        canvas_size=args.size,
        padding=args.padding
    )


if __name__ == "__main__":
    main()
