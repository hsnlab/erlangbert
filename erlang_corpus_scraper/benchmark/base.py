"""
Base evaluator interface for GraphCodeBERT evaluation tasks.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging
import torch

class BaseEvaluator(ABC):
    """Base class for all GraphCodeBERT evaluators."""
    
    def __init__(self, model_checkpoint: str, device: str = "auto"):
        """Initialize evaluator.
        
        Args:
            model_checkpoint: Path to model checkpoint
            device: Device to use ("auto", "cpu", "cuda")
        """
        self.model_checkpoint = model_checkpoint
        self.device = self._setup_device(device)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Model and tokenizer will be loaded lazily
        self.model = None
        self.tokenizer = None
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup computation device."""
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
                self.logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
            else:
                device = "cpu"
                self.logger.info("Using CPU")
        
        return torch.device(device)
    
    def _load_model_and_tokenizer(self):
        """Load model and tokenizer. Override in subclasses."""
        if self.model is not None and self.tokenizer is not None:
            return  # Already loaded
        
        self.logger.info(f"Loading model from {self.model_checkpoint}")
        # Subclasses should implement this
        raise NotImplementedError("Subclasses must implement _load_model_and_tokenizer")
    
    @abstractmethod
    def evaluate(self, data_path: str, **kwargs) -> Dict[str, float]:
        """Run evaluation and return metrics.
        
        Args:
            data_path: Path to evaluation data
            **kwargs: Additional evaluation parameters
            
        Returns:
            Dictionary of metric names to values
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Return evaluator name."""
        pass
    
    def cleanup(self):
        """Cleanup resources (models, GPU memory, etc.)."""
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            
        self.logger.info("Evaluator resources cleaned up")
