"""
Feature Extraction Module

Extracts features from grid squares including:
- Handwritten text detection using OCR
- Image detection and classification
- Question detection (text pattern matching)
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from PIL import Image
import numpy as np
import re

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from .grid_segmentation import GridSquare
from ..bayesian_feature_extractor import Attribute, AttributeType


@dataclass
class GridSquareFeatures:
    """Features extracted from a grid square."""
    square_id: str
    has_text: bool = False
    has_handwriting: bool = False
    has_image: bool = False
    has_question: bool = False
    text_confidence: float = 0.0
    image_complexity: float = 0.0
    question_type: Optional[str] = None
    extracted_text: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def to_attributes(self) -> List[Attribute]:
        """Convert features to BayesianFeatureExtractor Attributes."""
        attributes = [
            Attribute("has_text", AttributeType.BOOLEAN, value=self.has_text),
            Attribute("has_handwriting", AttributeType.BOOLEAN, value=self.has_handwriting),
            Attribute("has_image", AttributeType.BOOLEAN, value=self.has_image),
            Attribute("has_question", AttributeType.BOOLEAN, value=self.has_question),
            Attribute("text_confidence", AttributeType.NUMERICAL, value=self.text_confidence),
            Attribute("image_complexity", AttributeType.NUMERICAL, value=self.image_complexity),
        ]
        
        if self.question_type:
            attributes.append(
                Attribute("question_type", AttributeType.CATEGORICAL, value=self.question_type)
            )
        
        if self.extracted_text:
            attributes.append(
                Attribute("extracted_text", AttributeType.TEXT, value=self.extracted_text)
            )
        
        return attributes


class FeatureExtractor:
    """
    Extracts features from grid squares.
    
    Features include:
    - Text detection (OCR)
    - Handwriting detection
    - Image detection
    - Question pattern detection
    """
    
    def __init__(self):
        """Initialize feature extractor."""
        self.question_patterns = [
            r'\?',  # Question mark
            r'what|where|when|who|why|how',  # Question words
            r'select|choose|click|identify',  # Action questions
            r'which|all|none',  # Selection questions
        ]
        
        # Breadcrumb tracking for learning insights
        self.learning_breadcrumbs: List[Dict[str, Any]] = []
    
    def extract_features(self, grid_square: GridSquare) -> GridSquareFeatures:
        """
        Extract features from a grid square.
        
        Args:
            grid_square: GridSquare to extract features from
            
        Returns:
            GridSquareFeatures object
        """
        features = GridSquareFeatures(square_id=grid_square.square_id)
        
        # Convert PIL Image to numpy array for processing
        img_array = np.array(grid_square.image.convert('RGB'))
        
        # Detect text
        text_features = self._detect_text(grid_square.image)
        features.has_text = text_features['has_text']
        features.has_handwriting = text_features['has_handwriting']
        features.text_confidence = text_features['confidence']
        features.extracted_text = text_features['text']
        
        # Detect images
        image_features = self._detect_image(img_array)
        features.has_image = image_features['has_image']
        features.image_complexity = image_features['complexity']
        
        # Detect questions
        if features.extracted_text:
            question_features = self._detect_question(features.extracted_text)
            features.has_question = question_features['has_question']
            features.question_type = question_features['question_type']
        
        return features
    
    def _detect_text(self, image: Image.Image) -> Dict[str, Any]:
        """
        Detect text in image using OCR.
        
        Returns:
            Dictionary with text detection results
        """
        result = {
            'has_text': False,
            'has_handwriting': False,
            'confidence': 0.0,
            'text': None
        }
        
        if not TESSERACT_AVAILABLE:
            return result
        
        try:
            # Run OCR
            ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            
            # Extract text and confidence
            texts = [word for word in ocr_data['text'] if word.strip()]
            confidences = [float(conf) for conf in ocr_data['conf'] if conf != '-1']
            
            if texts and confidences:
                result['has_text'] = True
                result['confidence'] = np.mean(confidences) / 100.0  # Normalize to 0-1
                result['text'] = ' '.join(texts)
                
                # Simple heuristic for handwriting detection
                # Handwriting often has lower confidence and more variation
                if result['confidence'] < 0.7 and np.std(confidences) > 20:
                    result['has_handwriting'] = True
        except Exception as e:
            # OCR failed, no text detected
            pass
        
        return result
    
    def _detect_image(self, img_array: np.ndarray) -> Dict[str, Any]:
        """
        Detect if image contains significant visual content.
        
        Args:
            img_array: Image as numpy array
            
        Returns:
            Dictionary with image detection results
        """
        result = {
            'has_image': False,
            'complexity': 0.0
        }
        
        if not CV2_AVAILABLE:
            return result
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Calculate image complexity metrics
            # Variance indicates content richness
            variance = np.var(gray)
            
            # Edge density
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            # Combined complexity score
            complexity = (variance / 10000.0 + edge_density) / 2.0
            complexity = min(1.0, complexity)  # Cap at 1.0
            
            result['complexity'] = complexity
            result['has_image'] = complexity > 0.1  # Threshold for image content
        
        except Exception as e:
            pass
        
        return result
    
    def _detect_question(self, text: str) -> Dict[str, Any]:
        """
        Detect if text contains a question.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with question detection results
        """
        result = {
            'has_question': False,
            'question_type': None
        }
        
        text_lower = text.lower()
        
        # Check for question patterns
        for pattern in self.question_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                result['has_question'] = True
                
                # Classify question type
                if re.search(r'select|choose|click|identify|which|all|none', text_lower):
                    result['question_type'] = 'selection'
                elif re.search(r'what|where|when|who|why|how', text_lower):
                    result['question_type'] = 'wh_question'
                else:
                    result['question_type'] = 'general'
                
                break
        
        return result
    
    def extract_batch(self, grid_squares: List[GridSquare]) -> List[GridSquareFeatures]:
        """
        Extract features from multiple grid squares.
        
        Args:
            grid_squares: List of GridSquare objects
            
        Returns:
            List of GridSquareFeatures
        """
        return [self.extract_features(square) for square in grid_squares]
    
    def add_learning_breadcrumb(self, breadcrumb: Dict[str, Any]):
        """
        Add a learning breadcrumb from agents.
        
        Args:
            breadcrumb: Dictionary containing breadcrumb data with keys:
                - timestamp: ISO timestamp
                - event_type: Type of event
                - data: Event-specific data
                - source: Source of the breadcrumb
        """
        self.learning_breadcrumbs.append(breadcrumb)
        
        # Keep only last 1000 breadcrumbs
        if len(self.learning_breadcrumbs) > 1000:
            self.learning_breadcrumbs = self.learning_breadcrumbs[-1000:]
    
    def get_learning_breadcrumbs(self) -> List[Dict[str, Any]]:
        """Get all learning breadcrumbs."""
        return self.learning_breadcrumbs.copy()
    
    def get_breadcrumbs_by_type(self, event_type: str) -> List[Dict[str, Any]]:
        """Get breadcrumbs filtered by event type."""
        return [bc for bc in self.learning_breadcrumbs if bc.get("event_type") == event_type]