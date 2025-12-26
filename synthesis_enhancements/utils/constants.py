"""Constants for synthesis enhancements package."""

from enum import Enum
from typing import Final

# Operation types for classification
class OperationType(str, Enum):
    """Types of synthesis operations."""
    NOT_OPERATION = "NotOperation"
    STARTING_SYNTHESIS = "StartingSynthesis"
    MIXING = "Mixing"
    HEATING = "Heating"
    DRYING = "Drying"
    SHAPING = "Shaping"
    COOLING = "Cooling"
    QUENCHING = "Quenching"
    CALCINING = "Calcining"
    SINTERING = "Sintering"
    GRINDING = "Grinding"
    WASHING = "Washing"
    FILTERING = "Filtering"
    DISSOLVING = "Dissolving"


OPERATION_TYPES: Final[list[str]] = [op.value for op in OperationType]

# Material entity types
class MaterialEntityType(str, Enum):
    """Types of material entities."""
    TARGET = "Target"
    PRECURSOR = "Precursor"
    INTERMEDIATE = "Intermediate"
    DOPANT = "Dopant"
    SOLVENT = "Solvent"
    CATALYST = "Catalyst"
    OTHER = "Other"


MATERIAL_ENTITY_TYPES: Final[list[str]] = [met.value for met in MaterialEntityType]

# Processing defaults
DEFAULT_BATCH_SIZE: Final[int] = 32
DEFAULT_NUM_WORKERS: Final[int] = 4
DEFAULT_MAX_SEQUENCE_LENGTH: Final[int] = 512
DEFAULT_EMBEDDING_DIMENSION: Final[int] = 768

# Supported transformer models
SUPPORTED_MODELS: Final[dict[str, str]] = {
    "matbert": "m3rg-iitd/matscibert",
    "scibert": "allenai/scibert_scivocab_uncased",
    "bert-base": "bert-base-uncased",
    "roberta": "roberta-base",
    "pubmedbert": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
    "chembert": "seyonec/ChemBERTa-zinc-base-v1",
}

# Greek letter character range for material parsing
GREEK_LETTER_START: Final[int] = 945
GREEK_LETTER_END: Final[int] = 970

# Float precision for reaction balancing
FLOAT_ROUND_PRECISION: Final[int] = 3

# Model versioning
MODEL_REGISTRY_PATH: Final[str] = "mlruns"
DEFAULT_EXPERIMENT_NAME: Final[str] = "synthesis-extraction"

# ONNX serialization
ONNX_OPSET_VERSION: Final[int] = 14
