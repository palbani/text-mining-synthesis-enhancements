"""Configuration classes for synthesis enhancements."""

from dataclasses import dataclass, field
from typing import Optional, Literal
from pathlib import Path

from synthesis_enhancements.utils.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_NUM_WORKERS,
    DEFAULT_MAX_SEQUENCE_LENGTH,
    SUPPORTED_MODELS,
)


@dataclass
class ModelConfig:
    """Configuration for transformer models.

    Attributes:
        model_name: Name of the pretrained model (e.g., 'matbert', 'scibert')
        model_path: Optional path to local model weights
        max_sequence_length: Maximum input sequence length
        device: Device to run model on ('cuda', 'cpu', 'auto')
        use_fp16: Whether to use half-precision floating point
        cache_dir: Directory for caching downloaded models
    """
    model_name: str = "matbert"
    model_path: Optional[str] = None
    max_sequence_length: int = DEFAULT_MAX_SEQUENCE_LENGTH
    device: Literal["cuda", "cpu", "auto"] = "auto"
    use_fp16: bool = False
    cache_dir: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.model_name not in SUPPORTED_MODELS and self.model_path is None:
            raise ValueError(
                f"Unknown model '{self.model_name}'. "
                f"Supported models: {list(SUPPORTED_MODELS.keys())}. "
                "Alternatively, provide a custom model_path."
            )

    @property
    def hf_model_name(self) -> str:
        """Get HuggingFace model identifier."""
        if self.model_path:
            return self.model_path
        return SUPPORTED_MODELS.get(self.model_name, self.model_name)


@dataclass
class ProcessingConfig:
    """Configuration for parallel processing.

    Attributes:
        batch_size: Number of samples per batch
        num_workers: Number of parallel workers
        use_multiprocessing: Whether to use multiprocessing (vs threading)
        use_async: Whether to use async processing
        max_queue_size: Maximum size of processing queue
        timeout: Timeout in seconds for processing operations
        show_progress: Whether to show progress bar
    """
    batch_size: int = DEFAULT_BATCH_SIZE
    num_workers: int = DEFAULT_NUM_WORKERS
    use_multiprocessing: bool = True
    use_async: bool = False
    max_queue_size: int = 1000
    timeout: float = 300.0
    show_progress: bool = True

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if self.num_workers < 1:
            raise ValueError("num_workers must be at least 1")


@dataclass
class ExtractorConfig:
    """Configuration for operations and entity extraction.

    Attributes:
        model_config: Transformer model configuration
        processing_config: Parallel processing configuration
        spacy_model: SpaCy model to use for NLP
        confidence_threshold: Minimum confidence for predictions
        use_rule_refinement: Whether to apply rule-based refinement
        extract_conditions: Whether to extract synthesis conditions
    """
    model_config: ModelConfig = field(default_factory=ModelConfig)
    processing_config: ProcessingConfig = field(default_factory=ProcessingConfig)
    spacy_model: str = "en_core_web_sm"
    confidence_threshold: float = 0.5
    use_rule_refinement: bool = True
    extract_conditions: bool = True


@dataclass
class ParserConfig:
    """Configuration for material parsing.

    Attributes:
        model_config: Transformer model configuration for NER
        processing_config: Parallel processing configuration
        use_pubchem: Whether to validate against PubChem
        resolve_abbreviations: Whether to resolve material abbreviations
        decompose_mixtures: Whether to decompose mixtures into components
        resources_path: Path to parser resource files
    """
    model_config: ModelConfig = field(default_factory=ModelConfig)
    processing_config: ProcessingConfig = field(default_factory=ProcessingConfig)
    use_pubchem: bool = True
    resolve_abbreviations: bool = True
    decompose_mixtures: bool = True
    resources_path: Optional[Path] = None


@dataclass
class VersioningConfig:
    """Configuration for model versioning.

    Attributes:
        tracking_uri: MLflow tracking server URI
        experiment_name: Name of the MLflow experiment
        registry_uri: Model registry URI
        auto_log: Whether to auto-log metrics and parameters
    """
    tracking_uri: str = "mlruns"
    experiment_name: str = "synthesis-extraction"
    registry_uri: Optional[str] = None
    auto_log: bool = True


@dataclass
class SerializationConfig:
    """Configuration for ONNX serialization.

    Attributes:
        opset_version: ONNX opset version
        optimize: Whether to optimize the ONNX model
        quantize: Whether to quantize the model (int8)
        dynamic_axes: Whether to use dynamic axes for variable batch size
    """
    opset_version: int = 14
    optimize: bool = True
    quantize: bool = False
    dynamic_axes: bool = True
