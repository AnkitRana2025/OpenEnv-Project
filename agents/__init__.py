# agents/__init__.py - FIXED
from .baseline_agent import BaselineAgent
from .optimal_agent import OptimalAgent

# Comment out deepseek if not available
# from .deepseek_agent import DeepSeekAgent

__all__ = ['BaselineAgent', 'OptimalAgent']