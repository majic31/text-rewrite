import re
from typing import Dict, List, Tuple
from .base import BaseFilter

class RegexFilter(BaseFilter):
    """
    Regex Filter for pattern-based text replacements.
    Patterns are pre-compiled for performance.
    """
    def __init__(self, rules: Dict[str, str] = None, name: str = None):
        super().__init__(name=name)
        self.compiled_rules: List[Tuple[re.Pattern, str]] = []
        if rules:
            self.add_rules(rules)

    def add_rules(self, rules: Dict[str, str]):
        """
        Add new regex replacement rules.
        :param rules: A dictionary where key is the regex pattern and value is the replacement string.
        """
        for pattern, replacement in rules.items():
            compiled_pattern = re.compile(pattern)
            self.compiled_rules.append((compiled_pattern, replacement))

    def clear_rules(self):
        """
        Clear all existing regex rules.
        """
        self.compiled_rules.clear()

    def process(self, text: str) -> str:
        """
        Process text by sequentially applying all regex replacements.
        """
        if not text:
            return text
            
        current_text = text
        for compiled_pattern, replacement in self.compiled_rules:
            current_text = compiled_pattern.sub(replacement, current_text)
            
        return current_text
