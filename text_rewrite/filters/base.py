from abc import ABC, abstractmethod

class BaseFilter(ABC):
    """
    Abstract base class for all text filters.
    """
    def __init__(self, name: str = None):
        self.name = name or self.__class__.__name__

    @abstractmethod
    def process(self, text: str) -> str:
        """
        Process the input text and return the filtered text.
        """
        pass
