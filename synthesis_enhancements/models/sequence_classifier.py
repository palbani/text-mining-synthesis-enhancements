"""Sequence classification models for synthesis text mining."""

from typing import Any, Optional, Union
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    PreTrainedModel,
)

from synthesis_enhancements.models.base import (
    SynthesisTransformerModel,
    TransformerModelFactory,
)
from synthesis_enhancements.utils.config import ModelConfig
from synthesis_enhancements.utils.constants import OPERATION_TYPES

logger = logging.getLogger(__name__)


@TransformerModelFactory.register("operation_classifier")
class OperationClassifier(SynthesisTransformerModel):
    """Classifier for synthesis operation types.

    Classifies text segments into operation categories such as
    Heating, Mixing, Drying, etc.

    Attributes:
        num_labels: Number of operation classes
        label_names: Names of operation labels

    Example:
        >>> classifier = OperationClassifier(
        ...     config=ModelConfig(model_name="matbert"),
        ...     num_labels=7
        ... )
        >>> classifier.load()
        >>> result = classifier.predict("Heat the mixture at 500°C for 2 hours")
        >>> print(result.label)  # "Heating"
    """

    def __init__(
        self,
        config: ModelConfig,
        num_labels: int = 7,
        label_names: Optional[list[str]] = None,
    ) -> None:
        """Initialize the operation classifier.

        Args:
            config: Model configuration
            num_labels: Number of operation classes
            label_names: Optional list of label names
        """
        super().__init__(config)
        self.num_labels = num_labels
        self.label_names = label_names or OPERATION_TYPES[:num_labels]

    def _create_model(
        self, model_name: str, model_config: AutoConfig
    ) -> PreTrainedModel:
        """Create sequence classification model.

        Args:
            model_name: HuggingFace model name
            model_config: Model configuration

        Returns:
            Sequence classification model
        """
        model_config.num_labels = self.num_labels
        return AutoModelForSequenceClassification.from_pretrained(
            model_name,
            config=model_config,
            cache_dir=self.config.cache_dir,
        )

    def predict(
        self,
        texts: Union[str, list[str]],
        return_probabilities: bool = False,
        threshold: Optional[float] = None,
    ) -> Union[dict[str, Any], list[dict[str, Any]]]:
        """Classify operation type(s) for input text(s).

        Args:
            texts: Input text(s) to classify
            return_probabilities: Whether to return probability distribution
            threshold: Confidence threshold for prediction

        Returns:
            Classification result(s) with label, confidence, and optionally probabilities
        """
        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]

        # Encode inputs
        encoded = self.encode(texts)

        # Forward pass
        with torch.no_grad():
            outputs = self.model(**encoded)
            logits = outputs.logits
            probabilities = F.softmax(logits, dim=-1)

        # Process results
        results = []
        for i in range(len(texts)):
            probs = probabilities[i].cpu().numpy()
            pred_idx = probs.argmax()
            confidence = float(probs[pred_idx])

            result = {
                "label": self.label_names[pred_idx],
                "label_id": int(pred_idx),
                "confidence": confidence,
            }

            if return_probabilities:
                result["probabilities"] = {
                    label: float(probs[j])
                    for j, label in enumerate(self.label_names)
                }

            if threshold is not None and confidence < threshold:
                result["label"] = "Unknown"
                result["below_threshold"] = True

            results.append(result)

        return results[0] if single_input else results

    def predict_batch(
        self,
        texts: list[str],
        batch_size: int = 32,
        return_probabilities: bool = False,
    ) -> list[dict[str, Any]]:
        """Classify operation types in batches.

        Args:
            texts: List of input texts
            batch_size: Batch size for processing
            return_probabilities: Whether to return probability distribution

        Returns:
            List of classification results
        """
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_results = self.predict(batch, return_probabilities=return_probabilities)
            if isinstance(batch_results, dict):
                batch_results = [batch_results]
            results.extend(batch_results)
        return results


@TransformerModelFactory.register("paragraph_classifier")
class ParagraphClassifier(SynthesisTransformerModel):
    """Classifier for synthesis paragraph types.

    Classifies paragraphs as synthesis-relevant (solid-state, sol-gel, etc.)
    or non-synthesis content.

    Attributes:
        num_labels: Number of paragraph classes
        label_names: Names of paragraph labels

    Example:
        >>> classifier = ParagraphClassifier(
        ...     config=ModelConfig(model_name="scibert"),
        ...     num_labels=3,
        ...     label_names=["non_synthesis", "solid_state", "sol_gel"]
        ... )
        >>> classifier.load()
        >>> result = classifier.predict(paragraph_text)
    """

    DEFAULT_LABELS = ["non_synthesis", "solid_state", "sol_gel", "hydrothermal", "other"]

    def __init__(
        self,
        config: ModelConfig,
        num_labels: int = 3,
        label_names: Optional[list[str]] = None,
    ) -> None:
        """Initialize the paragraph classifier.

        Args:
            config: Model configuration
            num_labels: Number of paragraph classes
            label_names: Optional list of label names
        """
        super().__init__(config)
        self.num_labels = num_labels
        self.label_names = label_names or self.DEFAULT_LABELS[:num_labels]

    def _create_model(
        self, model_name: str, model_config: AutoConfig
    ) -> PreTrainedModel:
        """Create sequence classification model for paragraphs.

        Args:
            model_name: HuggingFace model name
            model_config: Model configuration

        Returns:
            Sequence classification model
        """
        model_config.num_labels = self.num_labels
        return AutoModelForSequenceClassification.from_pretrained(
            model_name,
            config=model_config,
            cache_dir=self.config.cache_dir,
        )

    def predict(
        self,
        texts: Union[str, list[str]],
        return_probabilities: bool = False,
        multi_label: bool = False,
    ) -> Union[dict[str, Any], list[dict[str, Any]]]:
        """Classify paragraph type(s) for input text(s).

        Args:
            texts: Input paragraph(s) to classify
            return_probabilities: Whether to return probability distribution
            multi_label: Whether to return multiple labels above threshold

        Returns:
            Classification result(s)
        """
        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]

        # Encode inputs
        encoded = self.encode(texts)

        # Forward pass
        with torch.no_grad():
            outputs = self.model(**encoded)
            logits = outputs.logits

            if multi_label:
                probabilities = torch.sigmoid(logits)
            else:
                probabilities = F.softmax(logits, dim=-1)

        # Process results
        results = []
        for i in range(len(texts)):
            probs = probabilities[i].cpu().numpy()
            pred_idx = probs.argmax()
            confidence = float(probs[pred_idx])

            result = {
                "label": self.label_names[pred_idx],
                "label_id": int(pred_idx),
                "confidence": confidence,
                "is_synthesis": self.label_names[pred_idx] != "non_synthesis",
            }

            if return_probabilities:
                result["probabilities"] = {
                    label: float(probs[j])
                    for j, label in enumerate(self.label_names)
                }

            if multi_label:
                # Return all labels above 0.5 threshold
                result["all_labels"] = [
                    self.label_names[j]
                    for j in range(len(self.label_names))
                    if probs[j] > 0.5
                ]

            results.append(result)

        return results[0] if single_input else results

    def filter_synthesis_paragraphs(
        self,
        paragraphs: list[str],
        threshold: float = 0.5,
    ) -> list[tuple[int, str, dict[str, Any]]]:
        """Filter paragraphs to keep only synthesis-relevant ones.

        Args:
            paragraphs: List of paragraph texts
            threshold: Confidence threshold for synthesis classification

        Returns:
            List of (index, text, result) tuples for synthesis paragraphs
        """
        results = self.predict(paragraphs, return_probabilities=True)
        if isinstance(results, dict):
            results = [results]

        synthesis_paragraphs = []
        for i, (text, result) in enumerate(zip(paragraphs, results)):
            if result["is_synthesis"] and result["confidence"] >= threshold:
                synthesis_paragraphs.append((i, text, result))

        return synthesis_paragraphs
