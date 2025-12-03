"""
Compile cropped truck entry images into 2000x2000 canvas images.
Stacks images side by side, top to bottom until canvas is full,
then creates a new canvas.
"""

import cv2
import os
import glob
import numpy as np
from datetime import datetime


def compile_crops(input_dir="debug_images", output_dir="compiled_canvases", canvas_size=2000, padding=5):
    """
    Compile all crop images from truck folders into canvas images.
    
    Args:
        input_dir: Directory containing Truck### folders
        output_dir: Directory to save compiled canvases
        canvas_size: Size of square canvas (default 2000x2000)
        padding: Pixels between images
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all crop images
    crop_files = []
    truck_folders = sorted(glob.glob(os.path.join(input_dir, "Truck*")))
    
    print(f"Found {len(truck_folders)} truck folders")
    
    for folder in truck_folders:
        # Look for the final crop image (entry_*_crop.jpg)
        crops = glob.glob(os.path.join(folder, "entry_*_crop.jpg"))
        if crops:
            # Take the first one (should only be one per truck)
            crop_files.append(crops[0])
    
    print(f"Found {len(crop_files)} crop images")
    
    if not crop_files:
        print("No crop images found!")
        return []
    
    # Load all images and track their sizes
    images = []
    for crop_file in crop_files:
        img = cv2.imread(crop_file)
        if img is not None:
            # Extract truck number from path
            truck_num = os.path.basename(os.path.dirname(crop_file))
            images.append({
                "image": img,
                "path": crop_file,
                "truck": truck_num,
                "height": img.shape[0],
                "width": img.shape[1]
            })
    
    print(f"Loaded {len(images)} images")
    
    # Calculate uniform height for all images (use median height)
    heights = [img["height"] for img in images]
    target_height = int(np.median(heights))
    
    # Cap target height to fit reasonable number of rows
    max_height = canvas_size // 4  # At least 4 rows
    target_height = min(target_height, max_height)
    
    print(f"Target row height: {target_height}px")
    
    # Resize all images to target height while maintaining aspect ratio
    for img_data in images:
        img = img_data["image"]
        h, w = img.shape[:2]
        scale = target_height / h
        new_width = int(w * scale)
        img_data["resized"] = cv2.resize(img, (new_width, target_height))
        img_data["resized_width"] = new_width
    
    # Pack images into canvases
    canvases = []
    canvas_num = 1
    
    # Current position on canvas
    x = padding
    y = padding
    row_height = target_height
    
    # Create first canvas
    canvas = np.ones((canvas_size, canvas_size, 3), dtype=np.uint8) * 40  # Dark gray background
    images_on_canvas = 0
    
    for img_data in images:
        resized = img_data["resized"]
        w = img_data["resized_width"]
        
        # Check if image fits in current row
        if x + w + padding > canvas_size:
            # Move to next row
            x = padding
            y += row_height + padding
        
        # Check if we need a new canvas
        if y + row_height + padding > canvas_size:
            # Save current canvas if it has images
            if images_on_canvas > 0:
                canvas_path = os.path.join(output_dir, f"canvas_{canvas_num:03d}.jpg")
                cv2.imwrite(canvas_path, canvas)
                print(f"Saved {canvas_path} with {images_on_canvas} images")
                canvases.append(canvas_path)
                canvas_num += 1
            
            # Create new canvas
            canvas = np.ones((canvas_size, canvas_size, 3), dtype=np.uint8) * 40
            x = padding
            y = padding
            images_on_canvas = 0
        
        # Place image on canvas
        canvas[y:y+row_height, x:x+w] = resized
        
        # Add truck label
        label = img_data["truck"].replace("Truck", "T")
        cv2.putText(canvas, label, (x + 5, y + 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        x += w + padding
        images_on_canvas += 1
    
    # Save final canvas if it has images
    if images_on_canvas > 0:
        canvas_path = os.path.join(output_dir, f"canvas_{canvas_num:03d}.jpg")
        cv2.imwrite(canvas_path, canvas)
        print(f"Saved {canvas_path} with {images_on_canvas} images")
        canvases.append(canvas_path)
    
    print(f"\n{'='*60}")
    print(f"COMPILATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total images compiled: {len(images)}")
    print(f"Total canvases created: {len(canvases)}")
    print(f"Output directory: {output_dir}/")
    print(f"{'='*60}")
    
    return canvases


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compile truck crop images into canvases")
    parser.add_argument("--input", "-i", default="debug_images", 
                        help="Input directory containing Truck### folders")
    parser.add_argument("--output", "-o", default="compiled_canvases",
                        help="Output directory for canvas images")
    parser.add_argument("--size", "-s", type=int, default=2000,
                        help="Canvas size in pixels (default: 2000)")
    parser.add_argument("--padding", "-p", type=int, default=5,
                        help="Padding between images (default: 5)")
    
    args = parser.parse_args()
    
    compile_crops(
        input_dir=args.input,
        output_dir=args.output,
        canvas_size=args.size,
        padding=args.padding
    )
