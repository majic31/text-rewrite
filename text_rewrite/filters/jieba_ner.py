import logging
from typing import Callable, Any
from .base import BaseFilter

logger = logging.getLogger(__name__)

class JiebaNERFilter(BaseFilter):
    """
    Extremely fast POS/NER Filter using Jieba.
    Uses `jieba.posseg` to perform POS tagging.
    """
    def __init__(self, replace_callback: Callable[[Any, str], str] = None, name: str = None):
        """
        :param replace_callback: A custom function that takes the jieba.posseg output (generator of pairs)
                                 and returns the modified text string.
        """
        super().__init__(name=name)
        self.replace_callback = replace_callback
        self._posseg = None
        self._loaded = False

    def _load_model(self):
        """
        Lazy load jieba to avoid overhead if not used.
        """
        if self._loaded:
            return
            
        logger.info("Lazy loading Jieba POS tagger...")
        try:
            import jieba.posseg as pseg
            self._posseg = pseg
            # Warm up jieba to load dictionary
            list(pseg.cut("预热"))
            self._loaded = True
            logger.info("Jieba loaded successfully.")
        except ImportError:
            raise ImportError("jieba is required for JiebaNERFilter. Run `pip install jieba`.")
        except Exception as e:
            logger.error(f"Failed to load jieba: {e}")
            raise

    def process(self, text: str) -> str:
        if not text:
            return text
            
        if not self._loaded:
            self._load_model()
            
        if not self._posseg:
            return text

        try:
            # pseg.cut returns a generator of pair(word, flag)
            # flag corresponds to POS tag, e.g., 'nr' for person name
            terms = list(self._posseg.cut(text, HMM=False))
            
            if self.replace_callback:
                return self.replace_callback(terms, text)
            else:
                return text
                
        except Exception as e:
            logger.error(f"Jieba NER processing failed: {e}")
            return text
