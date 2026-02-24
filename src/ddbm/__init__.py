"""
DDBM: Diophantine Dynamical Boundary Method
Statistical framework for chaos detection in time series

Version 7.2
"""

__version__ = "7.2.0"
__author__ = "Your Name"

from .core import analyze_timeseries, batch_analyze
from .config import DEFAULT_CONFIG

__all__ = ["analyze_timeseries", "batch_analyze", "DEFAULT_CONFIG"]
