"""
Benchmark module for GraphCodeBERT evaluation.

Provides evaluators for different downstream tasks:
- MLM: Masked Language Model evaluation  
- Code Search: Semantic code search evaluation (future)
- Clone Detection: Code clone detection evaluation (future)
"""

from .base import BaseEvaluator
from .mlm import MLMEvaluator

# Registry of available evaluators
EVALUATORS = {
    'mlm': MLMEvaluator,
    # Future evaluators:
    # 'code_search': CodeSearchEvaluator,
    # 'clone_detection': CloneDetectionEvaluator,
}

def get_evaluator(task_name: str):
    """Get evaluator class by task name.
    
    Args:
        task_name: Name of evaluation task ('mlm', 'code_search', etc.)
        
    Returns:
        Evaluator class
        
    Raises:
        ValueError: If task_name not recognized
    """
    if task_name not in EVALUATORS:
        available = ', '.join(EVALUATORS.keys())
        raise ValueError(f"Unknown evaluation task: {task_name}. Available: {available}")
    
    return EVALUATORS[task_name]

def list_evaluators():
    """List all available evaluator names."""
    return list(EVALUATORS.keys())

__all__ = ['BaseEvaluator', 'MLMEvaluator', 'get_evaluator', 'list_evaluators']
