"""Base classes for transformer models."""

from abc import ABC, abstractmethod
from typing import Any, Optional, Union
from pathlib import Path
import logging

import torch
import torch.nn as nn
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from synthesis_enhancements.utils.config import ModelConfig
from synthesis_enhancements.utils.constants import SUPPORTED_MODELS

logger = logging.getLogger(__name__)


class SynthesisTransformerModel(ABC, nn.Module):
    """Abstract base class for synthesis transformer models.

    This class provides common functionality for all transformer-based
    models used in synthesis text mining tasks.

    Attributes:
        config: Model configuration
        model: Underlying transformer model
        tokenizer: Model tokenizer
        device: Device model is running on
    """

    def __init__(self, config: ModelConfig) -> None:
        """Initialize the transformer model.

        Args:
            config: Model configuration object
        """
        super().__init__()
        self.config = config
        self._device: Optional[torch.device] = None
        self._model: Optional[PreTrainedModel] = None
        self._tokenizer: Optional[PreTrainedTokenizer] = None
        self._is_loaded = False

    @property
    def device(self) -> torch.device:
        """Get the device the model is running on."""
        if self._device is None:
            if self.config.device == "auto":
                self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self._device = torch.device(self.config.device)
        return self._device

    @property
    def model(self) -> PreTrainedModel:
        """Get the underlying transformer model."""
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self._model

    @property
    def tokenizer(self) -> PreTrainedTokenizer:
        """Get the model tokenizer."""
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not loaded. Call load() first.")
        return self._tokenizer

    def load(self) -> "SynthesisTransformerModel":
        """Load the model and tokenizer.

        Returns:
            Self for method chaining
        """
        if self._is_loaded:
            logger.warning("Model already loaded, skipping reload")
            return self

        model_name = self.config.hf_model_name
        logger.info(f"Loading model: {model_name}")

        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=self.config.cache_dir,
        )

        # Load model configuration
        model_config = AutoConfig.from_pretrained(
            model_name,
            cache_dir=self.config.cache_dir,
        )

        # Load model with task-specific head
        self._model = self._create_model(model_name, model_config)
        self._model.to(self.device)

        # Apply half precision if requested
        if self.config.use_fp16 and self.device.type == "cuda":
            self._model.half()

        self._is_loaded = True
        logger.info(f"Model loaded successfully on {self.device}")
        return self

    @abstractmethod
    def _create_model(
        self, model_name: str, model_config: AutoConfig
    ) -> PreTrainedModel:
        """Create the task-specific model.

        Args:
            model_name: HuggingFace model name
            model_config: Model configuration

        Returns:
            Initialized model
        """
        pass

    @abstractmethod
    def predict(
        self, texts: Union[str, list[str]], **kwargs: Any
    ) -> Union[Any, list[Any]]:
        """Make predictions on input texts.

        Args:
            texts: Input text(s) to process
            **kwargs: Additional prediction arguments

        Returns:
            Model predictions
        """
        pass

    def encode(
        self,
        texts: Union[str, list[str]],
        max_length: Optional[int] = None,
        return_tensors: str = "pt",
    ) -> dict[str, torch.Tensor]:
        """Encode texts using the tokenizer.

        Args:
            texts: Input text(s) to encode
            max_length: Maximum sequence length (uses config default if None)
            return_tensors: Return type ("pt" for PyTorch tensors)

        Returns:
            Dictionary of encoded inputs
        """
        if isinstance(texts, str):
            texts = [texts]

        max_length = max_length or self.config.max_sequence_length

        encoded = self.tokenizer(
            texts,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors=return_tensors,
        )

        return {k: v.to(self.device) for k, v in encoded.items()}

    def save(self, path: Union[str, Path]) -> None:
        """Save the model to disk.

        Args:
            path: Directory path to save model
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        logger.info(f"Model saved to {path}")

    @classmethod
    def from_pretrained(
        cls, path: Union[str, Path], config: Optional[ModelConfig] = None
    ) -> "SynthesisTransformerModel":
        """Load a model from a saved checkpoint.

        Args:
            path: Path to saved model
            config: Optional model configuration

        Returns:
            Loaded model instance
        """
        if config is None:
            config = ModelConfig(model_path=str(path))

        instance = cls(config)
        instance.load()
        return instance


class TransformerModelFactory:
    """Factory for creating transformer models.

    Provides a unified interface for instantiating different types of
    transformer models for synthesis text mining tasks.

    Example:
        >>> factory = TransformerModelFactory()
        >>> classifier = factory.create_classifier(
        ...     task="operation",
        ...     model_name="matbert",
        ...     num_labels=7
        ... )
        >>> classifier.load()
        >>> predictions = classifier.predict(["Heat the mixture at 500°C"])
    """

    _model_registry: dict[str, type[SynthesisTransformerModel]] = {}

    @classmethod
    def register(cls, task: str) -> callable:
        """Decorator to register a model class for a task.

        Args:
            task: Task name (e.g., 'operation_classifier', 'entity_recognizer')

        Returns:
            Decorator function
        """
        def decorator(model_cls: type[SynthesisTransformerModel]) -> type:
            cls._model_registry[task] = model_cls
            return model_cls
        return decorator

    @classmethod
    def create(
        cls,
        task: str,
        config: Optional[ModelConfig] = None,
        **kwargs: Any,
    ) -> SynthesisTransformerModel:
        """Create a model for the specified task.

        Args:
            task: Task name
            config: Model configuration
            **kwargs: Additional model arguments

        Returns:
            Initialized model (not yet loaded)

        Raises:
            ValueError: If task is not registered
        """
        if task not in cls._model_registry:
            available = list(cls._model_registry.keys())
            raise ValueError(
                f"Unknown task '{task}'. Available tasks: {available}"
            )

        model_cls = cls._model_registry[task]
        config = config or ModelConfig()
        return model_cls(config=config, **kwargs)

    @classmethod
    def create_classifier(
        cls,
        task: str = "operation",
        model_name: str = "matbert",
        num_labels: int = 7,
        **kwargs: Any,
    ) -> SynthesisTransformerModel:
        """Convenience method to create a classifier model.

        Args:
            task: Classification task ('operation' or 'paragraph')
            model_name: Name of pretrained model
            num_labels: Number of output labels
            **kwargs: Additional arguments

        Returns:
            Classifier model
        """
        config = ModelConfig(model_name=model_name)
        task_name = f"{task}_classifier"
        return cls.create(task_name, config=config, num_labels=num_labels, **kwargs)

    @classmethod
    def create_token_classifier(
        cls,
        task: str = "material_entity",
        model_name: str = "matbert",
        num_labels: int = 7,
        **kwargs: Any,
    ) -> SynthesisTransformerModel:
        """Convenience method to create a token classifier model.

        Args:
            task: Token classification task ('material_entity' or 'operation_token')
            model_name: Name of pretrained model
            num_labels: Number of entity labels
            **kwargs: Additional arguments

        Returns:
            Token classifier model
        """
        config = ModelConfig(model_name=model_name)
        task_name = f"{task}_recognizer"
        return cls.create(task_name, config=config, num_labels=num_labels, **kwargs)

    @classmethod
    def create_embedding_model(
        cls,
        model_name: str = "matbert",
        **kwargs: Any,
    ) -> SynthesisTransformerModel:
        """Convenience method to create an embedding model.

        Args:
            model_name: Name of pretrained model
            **kwargs: Additional arguments

        Returns:
            Embedding model
        """
        config = ModelConfig(model_name=model_name)
        return cls.create("embedding", config=config, **kwargs)

    @classmethod
    def list_available_tasks(cls) -> list[str]:
        """List all available tasks.

        Returns:
            List of registered task names
        """
        return list(cls._model_registry.keys())

    @classmethod
    def list_available_models(cls) -> list[str]:
        """List all supported pretrained models.

        Returns:
            List of supported model names
        """
        return list(SUPPORTED_MODELS.keys())
