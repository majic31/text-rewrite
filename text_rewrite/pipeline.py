from typing import List
import logging
from .filters.base import BaseFilter

logger = logging.getLogger(__name__)

class Pipeline:
    """
    TextRewrite Pipeline.
    Manages a sequence of filters and applies them to input text in order.
    """
    def __init__(self):
        self.filters: List[BaseFilter] = []

    def add_filter(self, text_filter: BaseFilter):
        """
        Register a filter to the pipeline.
        """
        if not isinstance(text_filter, BaseFilter):
            raise TypeError("Filter must be an instance of BaseFilter")
        self.filters.append(text_filter)
        return self # For method chaining

    def remove_filter(self, filter_name: str):
        """
        Remove a filter from the pipeline by its name.
        """
        original_length = len(self.filters)
        self.filters = [f for f in self.filters if f.name != filter_name]
        if len(self.filters) == original_length:
            logger.warning(f"Filter with name '{filter_name}' not found in pipeline.")
        return self
    def process(self, text: str) -> str:
        """
        Process the input text through all registered filters sequentially.
        """
        if not text:
            return text
            
        current_text = text
        for f in self.filters:
            try:
                current_text = f.process(current_text)
            except Exception as e:
                logger.error(f"Error processing text in filter {f.name}: {e}")
                # Depending on strictness, we might want to re-raise or continue.
                # For ASR/OCR robustness, continuing with the unmodified text of this step is often preferred.
                continue
                
        return current_text

    def __call__(self, text: str) -> str:
        return self.process(text)
