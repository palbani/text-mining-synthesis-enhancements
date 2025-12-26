"""
Synthesis Enhancements - Enhanced text-mining synthesis with transformer models.

This package provides modernized ML/NLP capabilities for extracting chemical
synthesis information from scientific literature, featuring:

- Transformer-based models (MatBERT, SciBERT)
- Parallel processing and multithreading
- Model versioning with MLflow
- ONNX model serialization
"""

__version__ = "1.0.0"
__author__ = "CederGroup Enhanced"

from synthesis_enhancements.models import (
    TransformerModelFactory,
    SynthesisTransformerModel,
    ModelConfig,
)
from synthesis_enhancements.processing import (
    ParallelProcessor,
    BatchProcessor,
    AsyncProcessor,
)
from synthesis_enhancements.versioning import ModelVersionManager
from synthesis_enhancements.serialization import ONNXSerializer
from synthesis_enhancements.extractors import (
    EnhancedOperationsExtractor,
    EnhancedMaterialEntityRecognizer,
)
from synthesis_enhancements.parsers import EnhancedMaterialParser

__all__ = [
    # Models
    "TransformerModelFactory",
    "SynthesisTransformerModel",
    "ModelConfig",
    # Processing
    "ParallelProcessor",
    "BatchProcessor",
    "AsyncProcessor",
    # Versioning
    "ModelVersionManager",
    # Serialization
    "ONNXSerializer",
    # Extractors
    "EnhancedOperationsExtractor",
    "EnhancedMaterialEntityRecognizer",
    # Parsers
    "EnhancedMaterialParser",
]
