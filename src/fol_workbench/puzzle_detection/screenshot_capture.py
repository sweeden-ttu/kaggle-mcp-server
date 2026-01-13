"""
Screenshot Capture Module for TorBrowser

Interface for capturing screenshots from TorBrowser with support for
window capture, full screen, or region selection.
"""

import platform
from typing import Optional, Tuple, Dict, Any
from pathlib import Path
from datetime import datetime
from PIL import Image
import numpy as np

try:
    import mss
    MSS_AVAILABLE = True
except ImportError:
    MSS_AVAILABLE = False

try:
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtGui import QScreen
    PYQT6_AVAILABLE = True
except ImportError:
    PYQT6_AVAILABLE = False


class ScreenshotCapture:
    """
    Captures screenshots from TorBrowser or system.
    
    Supports multiple capture methods:
    - Full screen
    - Window capture (by title/process)
    - Region selection
    """
    
    def __init__(self):
        """Initialize screenshot capture with available backends."""
        self.system = platform.system()
        self.metadata: Dict[str, Any] = {}
        
    def capture_full_screen(self) -> Image.Image:
        """
        Capture the full screen.
        
        Returns:
            PIL Image of the full screen
        """
        if MSS_AVAILABLE:
            return self._capture_with_mss()
        elif PYQT6_AVAILABLE:
            return self._capture_with_pyqt6()
        else:
            raise RuntimeError("No screenshot backend available. Install 'mss' or 'PyQt6'")
    
    def capture_window(self, window_title: Optional[str] = None, process_name: Optional[str] = None) -> Image.Image:
        """
        Capture a specific window.
        
        Args:
            window_title: Title of the window to capture
            process_name: Name of the process (e.g., 'firefox', 'tor')
            
        Returns:
            PIL Image of the window
        """
        # For now, fallback to full screen
        # In a full implementation, would use platform-specific window capture
        if window_title or process_name:
            # Try to find and capture specific window
            # This would require platform-specific code (X11, Windows API, etc.)
            pass
        
        return self.capture_full_screen()
    
    def capture_region(self, x: int, y: int, width: int, height: int) -> Image.Image:
        """
        Capture a specific region of the screen.
        
        Args:
            x: X coordinate of top-left corner
            y: Y coordinate of top-left corner
            width: Width of region
            height: Height of region
            
        Returns:
            PIL Image of the region
        """
        if MSS_AVAILABLE:
            with mss.mss() as sct:
                monitor = {
                    "top": y,
                    "left": x,
                    "width": width,
                    "height": height
                }
                screenshot = sct.grab(monitor)
                img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
                return img
        elif PYQT6_AVAILABLE:
            app = QApplication.instance()
            if app is None:
                app = QApplication([])
            
            screen = app.primaryScreen()
            pixmap = screen.grabWindow(0, x, y, width, height)
            img = pixmap.toImage()
            
            # Convert QImage to PIL Image
            width = img.width()
            height = img.height()
            ptr = img.bits()
            ptr.setsize(img.sizeInBytes())
            arr = np.array(ptr).reshape((height, width, 4))  # RGBA
            return Image.fromarray(arr[:, :, :3])  # Convert RGBA to RGB
        else:
            raise RuntimeError("No screenshot backend available")
    
    def capture_torbrowser(self) -> Image.Image:
        """
        Capture TorBrowser window specifically.
        
        Returns:
            PIL Image of TorBrowser window
        """
        # Try to find TorBrowser window
        # Common window titles: "Tor Browser", "The Tor Browser", etc.
        return self.capture_window(window_title="Tor Browser")
    
    def _capture_with_mss(self) -> Image.Image:
        """Capture using mss library."""
        with mss.mss() as sct:
            monitor = sct.monitors[1]  # Primary monitor
            screenshot = sct.grab(monitor)
            img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
            return img
    
    def _capture_with_pyqt6(self) -> Image.Image:
        """Capture using PyQt6."""
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        
        screen = app.primaryScreen()
        pixmap = screen.grabWindow(0)
        img = pixmap.toImage()
        
        # Convert QImage to PIL Image
        width = img.width()
        height = img.height()
        ptr = img.bits()
        ptr.setsize(img.sizeInBytes())
        arr = np.array(ptr).reshape((height, width, 4))  # RGBA
        return Image.fromarray(arr[:, :, :3])  # Convert RGBA to RGB
    
    def save_screenshot(self, image: Image.Image, filepath: Optional[Path] = None) -> Path:
        """
        Save screenshot to file with metadata.
        
        Args:
            image: PIL Image to save
            filepath: Optional path to save. If None, generates timestamped filename.
            
        Returns:
            Path to saved file
        """
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = Path(f"screenshot_{timestamp}.png")
        
        filepath = Path(filepath)
        image.save(filepath, "PNG")
        
        # Store metadata
        self.metadata = {
            "timestamp": datetime.now().isoformat(),
            "filepath": str(filepath),
            "size": image.size,
            "mode": image.mode
        }
        
        return filepath
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata from last capture."""
        return self.metadata.copy()
