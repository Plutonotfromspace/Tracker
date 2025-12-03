"""
Interactive tool to calibrate the detection zone polygon.
Click to add points, right-click to remove last point, press Enter to save.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import sys

import config


class ZoneCalibrator:
    def __init__(self):
        self.points = []
        self.frame = None
        self.fig = None
        self.ax = None
        self.polygon_patch = None
        self.point_plots = []
        
    def on_click(self, event):
        if event.inaxes != self.ax:
            return
            
        if event.button == 1:  # Left click - add point
            x, y = int(event.xdata), int(event.ydata)
            self.points.append([x, y])
            print(f"Point added: [{x}, {y}]")
            self.update_display()
        elif event.button == 3:  # Right click - remove last point
            if self.points:
                removed = self.points.pop()
                print(f"Point removed: {removed}")
                self.update_display()
                
    def on_key(self, event):
        if event.key == 'enter':
            self.save_and_quit()
        elif event.key == 'c':
            self.points = []
            print("Points cleared")
            self.update_display()
        elif event.key == 'escape':
            plt.close(self.fig)
            
    def update_display(self):
        # Clear previous drawings
        for p in self.point_plots:
            p.remove()
        self.point_plots = []
        
        if self.polygon_patch:
            self.polygon_patch.remove()
            self.polygon_patch = None
        
        # Draw points
        for i, pt in enumerate(self.points):
            circle = plt.Circle((pt[0], pt[1]), 8, color='lime', fill=True)
            self.ax.add_patch(circle)
            self.point_plots.append(circle)
            text = self.ax.text(pt[0] + 12, pt[1], str(i+1), color='lime', fontsize=10, fontweight='bold')
            self.point_plots.append(text)
        
        # Draw polygon if we have enough points
        if len(self.points) >= 3:
            polygon = patches.Polygon(self.points, fill=True, facecolor='lime', 
                                      edgecolor='lime', alpha=0.3, linewidth=2)
            self.ax.add_patch(polygon)
            self.polygon_patch = polygon
        elif len(self.points) == 2:
            line, = self.ax.plot([self.points[0][0], self.points[1][0]], 
                                 [self.points[0][1], self.points[1][1]], 
                                 'g-', linewidth=2)
            self.point_plots.append(line)
        
        self.fig.canvas.draw()
        
    def save_and_quit(self):
        if len(self.points) >= 3:
            print("\n" + "=" * 50)
            print("Copy this to config.py ZONE_POLYGON:")
            print("=" * 50)
            print(f"ZONE_POLYGON = {self.points}")
            print("=" * 50)
            
            # Auto-update config file
            try:
                with open('config.py', 'r') as f:
                    content = f.read()
                
                # Find and replace ZONE_POLYGON
                import re
                new_content = re.sub(
                    r'ZONE_POLYGON = \[[\s\S]*?\](?=\s*\n\s*#|\s*\n\s*\n|\s*$)',
                    f'ZONE_POLYGON = {self.points}',
                    content
                )
                
                with open('config.py', 'w') as f:
                    f.write(new_content)
                print("\n✓ config.py has been updated automatically!")
            except Exception as e:
                print(f"\nCould not auto-update config.py: {e}")
                print("Please copy the coordinates above manually.")
        else:
            print("\nNeed at least 3 points to define a zone!")
            
        plt.close(self.fig)
        
    def run(self, source=None):
        if source is None:
            source = config.VIDEO_SOURCE
            
        print(f"Opening video source: {source}")
        
        # Use imageio or opencv to read first frame
        try:
            import cv2
            cap = cv2.VideoCapture(source)
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                print("ERROR: Could not read frame from video")
                sys.exit(1)
                
            # Convert BGR to RGB for matplotlib
            self.frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"ERROR: {e}")
            sys.exit(1)
        
        print(f"Frame size: {self.frame.shape[1]}x{self.frame.shape[0]}")
        
        # Create matplotlib figure
        self.fig, self.ax = plt.subplots(1, 1, figsize=(14, 8))
        self.ax.imshow(self.frame)
        self.ax.set_title("Zone Calibrator\nLeft-click: Add point | Right-click: Remove point | Enter: Save | C: Clear | Esc: Cancel")
        
        # Draw existing zone from config in yellow
        if hasattr(config, 'ZONE_POLYGON') and config.ZONE_POLYGON:
            existing_polygon = patches.Polygon(config.ZONE_POLYGON, fill=False, 
                                               edgecolor='yellow', linewidth=2, linestyle='--')
            self.ax.add_patch(existing_polygon)
            self.ax.text(10, 30, "Yellow dashed = Current config zone", color='yellow', fontsize=10,
                        bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        
        # Connect events
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        plt.tight_layout()
        plt.show()
        
        if self.points:
            print(f"\nFinal zone polygon: {self.points}")


if __name__ == "__main__":
    calibrator = ZoneCalibrator()
    
    # Allow passing video source as argument
    source = sys.argv[1] if len(sys.argv) > 1 else None
    calibrator.run(source)
