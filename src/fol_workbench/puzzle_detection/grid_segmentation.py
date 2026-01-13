"""
Adaptive Grid Segmentation Module

Generates adaptive grid based on content detection with edge detection
and content-aware splitting.
"""

from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from PIL import Image
import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


@dataclass
class GridSquare:
    """Represents a single grid square with coordinates and image data."""
    x: int
    y: int
    width: int
    height: int
    image: Image.Image
    square_id: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.square_id is None:
            self.square_id = f"grid_{self.x}_{self.y}"


class GridSegmentation:
    """
    Adaptive grid generation based on content detection.
    
    Uses edge detection and content-aware splitting to create
    an adaptive grid that follows content boundaries.
    """
    
    def __init__(self, min_grid_size: int = 50, max_grid_size: int = 200):
        """
        Initialize grid segmentation.
        
        Args:
            min_grid_size: Minimum size of a grid square in pixels
            max_grid_size: Maximum size of a grid square in pixels
        """
        self.min_grid_size = min_grid_size
        self.max_grid_size = max_grid_size
    
    def segment_adaptive(self, image: Image.Image) -> List[GridSquare]:
        """
        Generate adaptive grid based on content detection.
        
        Args:
            image: PIL Image to segment
            
        Returns:
            List of GridSquare objects
        """
        if not CV2_AVAILABLE:
            # Fallback to uniform grid if OpenCV not available
            return self._create_uniform_grid(image)
        
        # Convert PIL to OpenCV format
        img_array = np.array(image.convert('RGB'))
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        # Detect edges
        edges = cv2.Canny(gray, 50, 150)
        
        # Detect horizontal and vertical lines
        horizontal_lines = self._detect_horizontal_lines(edges)
        vertical_lines = self._detect_vertical_lines(edges)
        
        # Create grid based on detected lines
        grid_squares = self._create_grid_from_lines(
            image, horizontal_lines, vertical_lines
        )
        
        # If no lines detected, use uniform grid
        if not grid_squares:
            grid_squares = self._create_uniform_grid(image)
        
        return grid_squares
    
    def segment_uniform(self, image: Image.Image, rows: int = 10, cols: int = 10) -> List[GridSquare]:
        """
        Generate uniform grid.
        
        Args:
            image: PIL Image to segment
            rows: Number of rows
            cols: Number of columns
            
        Returns:
            List of GridSquare objects
        """
        width, height = image.size
        cell_width = width // cols
        cell_height = height // rows
        
        grid_squares = []
        for row in range(rows):
            for col in range(cols):
                x = col * cell_width
                y = row * cell_height
                w = cell_width if col < cols - 1 else width - x
                h = cell_height if row < rows - 1 else height - y
                
                # Crop the square
                square_img = image.crop((x, y, x + w, y + h))
                
                grid_square = GridSquare(
                    x=x,
                    y=y,
                    width=w,
                    height=h,
                    image=square_img,
                    square_id=f"grid_{row}_{col}"
                )
                grid_squares.append(grid_square)
        
        return grid_squares
    
    def _detect_horizontal_lines(self, edges: np.ndarray) -> List[int]:
        """Detect horizontal lines in edge image."""
        # Use HoughLinesP or horizontal projection
        height, width = edges.shape
        
        # Horizontal projection
        horizontal_projection = np.sum(edges, axis=1)
        
        # Find peaks (potential horizontal lines)
        threshold = np.mean(horizontal_projection) * 1.5
        lines = []
        
        for y in range(height):
            if horizontal_projection[y] > threshold:
                # Check if this is a new line (not too close to existing ones)
                if not lines or abs(y - lines[-1]) > self.min_grid_size:
                    lines.append(y)
        
        return sorted(lines)
    
    def _detect_vertical_lines(self, edges: np.ndarray) -> List[int]:
        """Detect vertical lines in edge image."""
        height, width = edges.shape
        
        # Vertical projection
        vertical_projection = np.sum(edges, axis=0)
        
        # Find peaks (potential vertical lines)
        threshold = np.mean(vertical_projection) * 1.5
        lines = []
        
        for x in range(width):
            if vertical_projection[x] > threshold:
                # Check if this is a new line (not too close to existing ones)
                if not lines or abs(x - lines[-1]) > self.min_grid_size:
                    lines.append(x)
        
        return sorted(lines)
    
    def _create_grid_from_lines(
        self,
        image: Image.Image,
        horizontal_lines: List[int],
        vertical_lines: List[int]
    ) -> List[GridSquare]:
        """Create grid squares from detected lines."""
        width, height = image.size
        
        # Add boundaries
        if not horizontal_lines or horizontal_lines[0] > 0:
            horizontal_lines.insert(0, 0)
        if not horizontal_lines or horizontal_lines[-1] < height:
            horizontal_lines.append(height)
        
        if not vertical_lines or vertical_lines[0] > 0:
            vertical_lines.insert(0, 0)
        if not vertical_lines or vertical_lines[-1] < width:
            vertical_lines.append(width)
        
        grid_squares = []
        square_id = 0
        
        for i in range(len(horizontal_lines) - 1):
            for j in range(len(vertical_lines) - 1):
                x = vertical_lines[j]
                y = horizontal_lines[i]
                w = vertical_lines[j + 1] - x
                h = horizontal_lines[i + 1] - y
                
                # Skip if too small
                if w < self.min_grid_size or h < self.min_grid_size:
                    continue
                
                # Crop the square
                square_img = image.crop((x, y, x + w, y + h))
                
                grid_square = GridSquare(
                    x=x,
                    y=y,
                    width=w,
                    height=h,
                    image=square_img,
                    square_id=f"grid_{square_id}"
                )
                grid_squares.append(grid_square)
                square_id += 1
        
        return grid_squares
    
    def _create_uniform_grid(self, image: Image.Image) -> List[GridSquare]:
        """Create uniform grid as fallback."""
        width, height = image.size
        
        # Calculate grid size
        grid_size = max(self.min_grid_size, min(self.max_grid_size, width // 10, height // 10))
        
        rows = (height + grid_size - 1) // grid_size
        cols = (width + grid_size - 1) // grid_size
        
        return self.segment_uniform(image, rows, cols)
