import logging
from typing import Callable, Any
from .base import BaseFilter

logger = logging.getLogger(__name__)

class NERFilter(BaseFilter):
    """
    NER Filter for replacing or normalizing entities based on Named Entity Recognition.
    Uses HanLP. Supports lazy loading of the model to save memory until first use.
    """
    def __init__(self, name: str = "NERFilter", model_name: str = "LARGE_ALBERT_BASE", replace_callback: Callable[[Any], str] = None):
        """
        :param model_name: HanLP model identifier for NER/POS.
        :param replace_callback: A custom function that takes the HanLP output (e.g., list of words/tags)
                                 and returns the modified text string.
        """
        super().__init__(name=name)
        self.model_name = model_name
        self.replace_callback = replace_callback
        self._model = None
        self._loaded = False

    def _load_model(self):
        """
        Lazy load the HanLP model.
        """
        if self._loaded:
            return
            
        logger.info(f"Lazy loading HanLP NER model '{self.model_name}'...")
        try:
            import hanlp
            # In a real scenario, you'd load the specific NER/POS model you need.
            # Example: tok/pos/ner pipeline
            self._model = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_SMALL_ZH)
            self._loaded = True
            logger.info("HanLP NER model loaded successfully.")
        except ImportError:
            raise ImportError("hanlp is required for NERFilter. Run `pip install hanlp`.")
        except Exception as e:
            logger.error(f"Failed to load HanLP model: {e}")
            raise

    def process(self, text: str) -> str:
        """
        Process text by extracting entities and applying the replacement logic.
        """
        if not text:
            return text
            
        if not self._loaded:
            self._load_model()
            
        if not self._model:
            return text # Fail-safe, return original if model failed to load but no exception was raised

        try:
            # Note: actual HanLP return format depends on the specific model loaded.
            # Here we assume a MTL (multi-task learning) model that returns a dict.
            doc = self._model(text)
            
            if self.replace_callback:
                return self.replace_callback(doc, text)
            else:
                # Default behavior: do nothing if no callback is provided, or implement a basic rule
                # (e.g., wrap entities in brackets). Here we just return the text.
                return text
                
        except Exception as e:
            logger.error(f"NER processing failed: {e}")
            return text
