from typing import Dict, List
from .base import BaseFilter

try:
    from flashtext import KeywordProcessor
except ImportError:
    KeywordProcessor = None


class HotwordFilter(BaseFilter):
    """
    Hotword Filter for high-performance exact word replacements.
    Uses Aho-Corasick algorithm via flashtext library.
    Ideal for massive dictionaries (100k+ words).
    """
    def __init__(self, hotwords: Dict[str, str] = None, case_sensitive: bool = False, name: str = None):
        super().__init__(name=name)
        if KeywordProcessor is None:
            raise ImportError("flashtext library is required for HotwordFilter. Run `pip install flashtext`.")
            
        self.keyword_processor = KeywordProcessor(case_sensitive=case_sensitive)
        if hotwords:
            self.add_hotwords(hotwords)

    def add_hotwords(self, hotwords: Dict[str, str]):
        """
        Add hotword replacements dynamically.
        :param hotwords: Dictionary mapping original words to target replacement words.
                         Note: flashtext format is { 'replacement': ['word1', 'word2'] } for add_keywords_from_dict.
                         Since we get { 'word': 'replacement' }, we iterate or transform it.
        """
        for original_word, replacement_word in hotwords.items():
            self.keyword_processor.add_keyword(original_word, replacement_word)

    def remove_hotwords(self, words: List[str]):
        """
        Remove hotwords from the trie dynamically.
        """
        for word in words:
            self.keyword_processor.remove_keyword(word)

    def process(self, text: str) -> str:
        """
        Process text by efficiently finding and replacing all matching hotwords.
        """
        if not text:
            return text
            
        # flashtext operates in O(N) where N is text length, extremely fast.
        return self.keyword_processor.replace_keywords(text)
