"""
Autocomplete and Suggestion System for FOL Workbench

Provides predictive suggestions as users type formulas, learns from user behavior,
and tracks preferences for optimization.
"""

import json
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime


@dataclass
class Suggestion:
    """Represents a suggestion for autocomplete."""
    text: str
    display: str
    category: str
    confidence: float
    usage_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Suggestion':
        return cls(**data)


@dataclass
class UserPreference:
    """User preference data."""
    font_size: int = 12
    font_family: str = "Consolas"
    preferred_suggestions: List[str] = None
    ignored_suggestions: List[str] = None
    click_patterns: Dict[str, int] = None
    
    def __post_init__(self):
        if self.preferred_suggestions is None:
            self.preferred_suggestions = []
        if self.ignored_suggestions is None:
            self.ignored_suggestions = []
        if self.click_patterns is None:
            self.click_patterns = {}
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserPreference':
        return cls(**data)


class SuggestionEngine:
    """Engine that generates and learns from suggestions."""
    
    # Common FOL patterns and keywords
    KEYWORDS = [
        "And", "Or", "Not", "Implies", "Iff", "ForAll", "Exists",
        "True", "False", "x", "y", "z", "w", "a", "b", "c"
    ]
    
    # Common formula patterns
    PATTERNS = [
        "And({var}, {var})",
        "Or({var}, Not({var}))",
        "Implies({var}, {var})",
        "Iff({var}, {var})",
        "And(Or({var}, {var}), {var})",
        "Not(And({var}, {var}))",
        "Or({var}, Not({var}))",
    ]
    
    def __init__(self, preferences_path: Optional[Path] = None):
        self.preferences_path = preferences_path or Path("user_preferences.json")
        self.preferences = self._load_preferences()
        
        # Track usage statistics
        self.usage_stats: Dict[str, int] = defaultdict(int)
        self.click_stats: Dict[str, int] = defaultdict(int)
        self.ignored_suggestions: set = set(self.preferences.ignored_suggestions)
        self.preferred_suggestions: set = set(self.preferences.preferred_suggestions)
        
        # Character count tracking for auto-submit
        self.last_char_count = 0
        self.current_input = ""
    
    def _load_preferences(self) -> UserPreference:
        """Load user preferences from file."""
        if self.preferences_path.exists():
            try:
                with open(self.preferences_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return UserPreference.from_dict(data)
            except Exception:
                pass
        return UserPreference()
    
    def _save_preferences(self):
        """Save user preferences to file."""
        try:
            self.preferences.preferred_suggestions = list(self.preferred_suggestions)
            self.preferences.ignored_suggestions = list(self.ignored_suggestions)
            self.preferences.click_patterns = dict(self.click_stats)
            
            with open(self.preferences_path, 'w', encoding='utf-8') as f:
                json.dump(self.preferences.to_dict(), f, indent=2, ensure_ascii=False)
        except Exception:
            pass
    
    def generate_suggestions(self, input_text: str, max_suggestions: int = 10) -> List[Suggestion]:
        """
        Generate suggestions based on current input.
        
        Args:
            input_text: Current text being typed
            max_suggestions: Maximum number of suggestions to return
            
        Returns:
            List of suggestions sorted by relevance
        """
        if not input_text:
            return self._get_default_suggestions(max_suggestions)
        
        input_lower = input_text.lower()
        suggestions = []
        
        # 1. Keyword matches
        for keyword in self.KEYWORDS:
            if keyword.lower().startswith(input_lower):
                confidence = self._calculate_confidence(keyword, input_text)
                suggestions.append(Suggestion(
                    text=keyword,
                    display=keyword,
                    category="Keyword",
                    confidence=confidence,
                    usage_count=self.usage_stats.get(keyword, 0)
                ))
        
        # 2. Pattern completions
        if input_text.endswith("(") or input_text.endswith(","):
            # Suggest variable names or nested patterns
            for var in ["x", "y", "z", "w", "a", "b", "c"]:
                if var not in input_text.lower():
                    suggestions.append(Suggestion(
                        text=var,
                        display=var,
                        category="Variable",
                        confidence=0.7,
                        usage_count=self.usage_stats.get(var, 0)
                    ))
        
        # 3. Formula pattern suggestions
        if len(input_text) >= 2:
            for pattern in self.PATTERNS:
                # Try to match pattern
                if self._matches_pattern(input_text, pattern):
                    completion = self._complete_pattern(input_text, pattern)
                    if completion:
                        suggestions.append(Suggestion(
                            text=completion,
                            display=completion,
                            category="Pattern",
                            confidence=0.6,
                            usage_count=self.usage_stats.get(completion, 0)
                        ))
        
        # 4. Learned suggestions from usage
        for suggestion_text in self.preferred_suggestions:
            if suggestion_text.lower().startswith(input_lower):
                suggestions.append(Suggestion(
                    text=suggestion_text,
                    display=suggestion_text,
                    category="Learned",
                    confidence=0.9,  # High confidence for learned patterns
                    usage_count=self.usage_stats.get(suggestion_text, 0)
                ))
        
        # Filter out ignored suggestions
        suggestions = [s for s in suggestions if s.text not in self.ignored_suggestions]
        
        # Sort by confidence and usage
        suggestions.sort(key=lambda s: (s.confidence, s.usage_count), reverse=True)
        
        return suggestions[:max_suggestions]
    
    def _get_default_suggestions(self, max_suggestions: int) -> List[Suggestion]:
        """Get default suggestions when input is empty."""
        defaults = [
            "And(x, y)",
            "Or(x, Not(y))",
            "Implies(x, y)",
            "Iff(x, y)",
            "Not(And(x, y))",
        ]
        
        suggestions = []
        for text in defaults[:max_suggestions]:
            suggestions.append(Suggestion(
                text=text,
                display=text,
                category="Default",
                confidence=0.5,
                usage_count=self.usage_stats.get(text, 0)
            ))
        
        return suggestions
    
    def _calculate_confidence(self, keyword: str, input_text: str) -> float:
        """Calculate confidence score for a suggestion."""
        base_confidence = 0.8
        
        # Boost if it's a preferred suggestion
        if keyword in self.preferred_suggestions:
            base_confidence += 0.1
        
        # Boost based on usage
        usage_count = self.usage_stats.get(keyword, 0)
        if usage_count > 0:
            base_confidence += min(0.1, usage_count / 100.0)
        
        # Penalize if it's been ignored
        if keyword in self.ignored_suggestions:
            base_confidence -= 0.3
        
        return max(0.0, min(1.0, base_confidence))
    
    def _matches_pattern(self, input_text: str, pattern: str) -> bool:
        """Check if input text matches a pattern."""
        # Simple pattern matching - check if structure is similar
        if "And" in pattern and "And" in input_text:
            return True
        if "Or" in pattern and "Or" in input_text:
            return True
        if "Implies" in pattern and "Implies" in input_text:
            return True
        return False
    
    def _complete_pattern(self, input_text: str, pattern: str) -> Optional[str]:
        """Try to complete a pattern based on input."""
        # Extract variables already used
        used_vars = set(re.findall(r'\b([a-z])\b', input_text.lower()))
        available_vars = [v for v in ["x", "y", "z", "w"] if v not in used_vars]
        
        if not available_vars:
            return None
        
        # Try to complete the pattern
        if "And(" in input_text and not input_text.endswith(")"):
            if "," not in input_text[-10:]:  # No comma recently
                return f"{input_text}, {available_vars[0]})"
            else:
                return f"{input_text})"
        
        return None
    
    def track_usage(self, text: str):
        """Track that a suggestion or formula was used."""
        self.usage_stats[text] += 1
        self._save_preferences()
    
    def track_click(self, suggestion_text: str, event_type: str = "click"):
        """Track user click/interaction with a suggestion."""
        key = f"{event_type}:{suggestion_text}"
        self.click_stats[key] += 1
        
        # Mark as preferred if clicked multiple times
        if self.click_stats[key] >= 3:
            self.preferred_suggestions.add(suggestion_text)
        
        self._save_preferences()
    
    def ignore_suggestion(self, suggestion_text: str):
        """Mark a suggestion as ignored."""
        self.ignored_suggestions.add(suggestion_text)
        if suggestion_text in self.preferred_suggestions:
            self.preferred_suggestions.remove(suggestion_text)
        self._save_preferences()
    
    def update_font_preference(self, font_family: str, font_size: int):
        """Update user's font preferences."""
        self.preferences.font_family = font_family
        self.preferences.font_size = font_size
        self._save_preferences()
    
    def should_auto_submit(self, current_text: str) -> bool:
        """
        Check if we should auto-submit based on character count.
        Returns True when character count is divisible by 3.
        """
        char_count = len(current_text)
        self.current_input = current_text
        
        # Check if count is divisible by 3 and has changed
        if char_count > 0 and char_count % 3 == 0 and char_count != self.last_char_count:
            self.last_char_count = char_count
            return True
        
        return False
    
    def get_analytics(self) -> Dict[str, Any]:
        """Get analytics data about usage."""
        return {
            "total_suggestions_used": sum(self.usage_stats.values()),
            "most_used": dict(Counter(self.usage_stats).most_common(10)),
            "click_patterns": dict(self.click_stats),
            "preferred_count": len(self.preferred_suggestions),
            "ignored_count": len(self.ignored_suggestions),
            "font_preferences": {
                "family": self.preferences.font_family,
                "size": self.preferences.font_size
            }
        }
