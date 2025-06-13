"""
Drift Detection Module for Book Recommendation System

This module provides comprehensive drift detection capabilities for monitoring
changes in input queries and output recommendations in the Ukrainian book
recommendation system.

Key Components:
- DriftDetector: Core drift detection algorithms
- DriftReporter: Comprehensive reporting and insights
- DriftMetrics: Data structure for drift measurements

Features:
- Input drift detection (semantic, statistical, vocabulary, language patterns)
- Output drift detection (genre distribution, popularity, diversity, semantics)
- Ukrainian language support
- Actionable insights and recommendations
- Integration with MLOps monitoring stack
"""

from .detector import DriftDetector, DriftReporter, DriftMetrics

__version__ = "1.0.0"
__author__ = "MLOps Team"

__all__ = [
    "DriftDetector",
    "DriftReporter", 
    "DriftMetrics"
]

# Default configuration for Ukrainian book recommendation system
DEFAULT_CONFIG = {
    "statistical_threshold": 0.05,      # P-value for statistical tests
    "semantic_threshold": 0.8,          # Cosine similarity threshold
    "distribution_threshold": 0.1,      # Jensen-Shannon divergence threshold
    "language_threshold": 0.2,          # Ukrainian/English ratio threshold
    "diversity_threshold": 0.2,         # Recommendation diversity threshold
    "popularity_threshold": 0.6         # Book popularity overlap threshold
}

def create_detector(config=None):
    """
    Create a DriftDetector with default or custom configuration.
    
    Args:
        config (dict, optional): Custom configuration parameters
        
    Returns:
        DriftDetector: Configured drift detector instance
    """
    if config is None:
        config = DEFAULT_CONFIG
    
    return DriftDetector(
        statistical_threshold=config.get("statistical_threshold", 0.05),
        semantic_threshold=config.get("semantic_threshold", 0.8),
        distribution_threshold=config.get("distribution_threshold", 0.1)
    )

def create_reporter():
    """
    Create a DriftReporter instance.
    
    Returns:
        DriftReporter: Drift reporter instance
    """
    return DriftReporter()