"""Token classification models for synthesis text mining."""

from typing import Any, Optional, Union
from dataclasses import dataclass
import logging

import torch
import torch.nn.functional as F
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    PreTrainedModel,
)

from synthesis_enhancements.models.base import (
    SynthesisTransformerModel,
    TransformerModelFactory,
)
from synthesis_enhancements.utils.config import ModelConfig
from synthesis_enhancements.utils.constants import MATERIAL_ENTITY_TYPES

logger = logging.getLogger(__name__)


@dataclass
class Entity:
    """Represents a recognized entity.

    Attributes:
        text: The entity text
        label: Entity label/type
        start: Start character position
        end: End character position
        confidence: Prediction confidence score
    """
    text: str
    label: str
    start: int
    end: int
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        """Convert entity to dictionary."""
        return {
            "text": self.text,
            "label": self.label,
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence,
        }


@TransformerModelFactory.register("material_entity_recognizer")
class MaterialEntityRecognizer(SynthesisTransformerModel):
    """Named entity recognizer for material entities.

    Identifies and classifies material mentions in synthesis text,
    including targets, precursors, dopants, solvents, etc.

    Example:
        >>> recognizer = MaterialEntityRecognizer(
        ...     config=ModelConfig(model_name="matbert"),
        ...     num_labels=15  # BIO tagging for 7 entity types + O
        ... )
        >>> recognizer.load()
        >>> entities = recognizer.predict("LiCoO2 was synthesized from Li2CO3 and Co3O4")
        >>> for entity in entities:
        ...     print(f"{entity.text}: {entity.label}")
    """

    # BIO tagging scheme labels
    DEFAULT_LABELS = [
        "O",
        "B-Target", "I-Target",
        "B-Precursor", "I-Precursor",
        "B-Intermediate", "I-Intermediate",
        "B-Dopant", "I-Dopant",
        "B-Solvent", "I-Solvent",
        "B-Catalyst", "I-Catalyst",
        "B-Other", "I-Other",
    ]

    def __init__(
        self,
        config: ModelConfig,
        num_labels: int = 15,
        label_names: Optional[list[str]] = None,
    ) -> None:
        """Initialize the material entity recognizer.

        Args:
            config: Model configuration
            num_labels: Number of BIO labels
            label_names: Optional list of label names
        """
        super().__init__(config)
        self.num_labels = num_labels
        self.label_names = label_names or self.DEFAULT_LABELS[:num_labels]

        # Build label mappings
        self.label2id = {label: i for i, label in enumerate(self.label_names)}
        self.id2label = {i: label for i, label in enumerate(self.label_names)}

    def _create_model(
        self, model_name: str, model_config: AutoConfig
    ) -> PreTrainedModel:
        """Create token classification model.

        Args:
            model_name: HuggingFace model name
            model_config: Model configuration

        Returns:
            Token classification model
        """
        model_config.num_labels = self.num_labels
        model_config.id2label = self.id2label
        model_config.label2id = self.label2id
        return AutoModelForTokenClassification.from_pretrained(
            model_name,
            config=model_config,
            cache_dir=self.config.cache_dir,
        )

    def predict(
        self,
        texts: Union[str, list[str]],
        aggregate_entities: bool = True,
    ) -> Union[list[Entity], list[list[Entity]]]:
        """Recognize material entities in text(s).

        Args:
            texts: Input text(s)
            aggregate_entities: Whether to aggregate BIO tags into entities

        Returns:
            List of recognized entities (or list of lists for batch input)
        """
        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]

        # Encode inputs
        encoded = self.tokenizer(
            texts,
            max_length=self.config.max_sequence_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
            return_offsets_mapping=True,
        )

        offset_mapping = encoded.pop("offset_mapping")
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        # Forward pass
        with torch.no_grad():
            outputs = self.model(**encoded)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            probabilities = F.softmax(logits, dim=-1)

        # Process results
        all_entities = []
        for i in range(len(texts)):
            text = texts[i]
            preds = predictions[i].cpu().numpy()
            probs = probabilities[i].cpu().numpy()
            offsets = offset_mapping[i].numpy()

            if aggregate_entities:
                entities = self._aggregate_entities(text, preds, probs, offsets)
            else:
                entities = self._get_token_predictions(text, preds, probs, offsets)

            all_entities.append(entities)

        return all_entities[0] if single_input else all_entities

    def _aggregate_entities(
        self,
        text: str,
        predictions: Any,
        probabilities: Any,
        offsets: Any,
    ) -> list[Entity]:
        """Aggregate BIO tags into entity spans.

        Args:
            text: Original text
            predictions: Token predictions
            probabilities: Token probabilities
            offsets: Character offset mapping

        Returns:
            List of aggregated entities
        """
        entities = []
        current_entity = None

        for idx, (pred, probs, (start, end)) in enumerate(
            zip(predictions, probabilities, offsets)
        ):
            # Skip special tokens (offset 0,0)
            if start == end == 0:
                continue

            label = self.id2label[pred]
            confidence = float(probs[pred])

            if label.startswith("B-"):
                # Save previous entity if exists
                if current_entity is not None:
                    entities.append(current_entity)

                # Start new entity
                entity_type = label[2:]
                current_entity = Entity(
                    text=text[start:end],
                    label=entity_type,
                    start=int(start),
                    end=int(end),
                    confidence=confidence,
                )

            elif label.startswith("I-") and current_entity is not None:
                entity_type = label[2:]
                if entity_type == current_entity.label:
                    # Extend current entity
                    current_entity.text = text[current_entity.start : end]
                    current_entity.end = int(end)
                    current_entity.confidence = min(
                        current_entity.confidence, confidence
                    )

            else:  # O label or mismatched I- tag
                if current_entity is not None:
                    entities.append(current_entity)
                    current_entity = None

        # Add final entity if exists
        if current_entity is not None:
            entities.append(current_entity)

        return entities

    def _get_token_predictions(
        self,
        text: str,
        predictions: Any,
        probabilities: Any,
        offsets: Any,
    ) -> list[dict[str, Any]]:
        """Get per-token predictions without aggregation.

        Args:
            text: Original text
            predictions: Token predictions
            probabilities: Token probabilities
            offsets: Character offset mapping

        Returns:
            List of token-level predictions
        """
        token_preds = []
        for pred, probs, (start, end) in zip(predictions, probabilities, offsets):
            if start == end == 0:
                continue

            token_preds.append({
                "text": text[start:end],
                "label": self.id2label[pred],
                "start": int(start),
                "end": int(end),
                "confidence": float(probs[pred]),
            })

        return token_preds

    def extract_materials_by_role(
        self, text: str
    ) -> dict[str, list[Entity]]:
        """Extract materials grouped by their role.

        Args:
            text: Input text

        Returns:
            Dictionary mapping roles to lists of entities
        """
        entities = self.predict(text)
        by_role: dict[str, list[Entity]] = {}

        for entity in entities:
            role = entity.label
            if role not in by_role:
                by_role[role] = []
            by_role[role].append(entity)

        return by_role


@TransformerModelFactory.register("operation_token_classifier")
class OperationTokenClassifier(SynthesisTransformerModel):
    """Token classifier for synthesis operations.

    Identifies operation tokens within sentences and classifies
    them into operation types.

    Example:
        >>> classifier = OperationTokenClassifier(
        ...     config=ModelConfig(model_name="matbert"),
        ...     num_labels=15
        ... )
        >>> classifier.load()
        >>> operations = classifier.predict("The mixture was heated and then dried")
    """

    DEFAULT_LABELS = [
        "O",
        "B-Heating", "I-Heating",
        "B-Mixing", "I-Mixing",
        "B-Drying", "I-Drying",
        "B-Shaping", "I-Shaping",
        "B-Cooling", "I-Cooling",
        "B-Grinding", "I-Grinding",
        "B-Other", "I-Other",
    ]

    def __init__(
        self,
        config: ModelConfig,
        num_labels: int = 15,
        label_names: Optional[list[str]] = None,
    ) -> None:
        """Initialize the operation token classifier.

        Args:
            config: Model configuration
            num_labels: Number of BIO labels
            label_names: Optional list of label names
        """
        super().__init__(config)
        self.num_labels = num_labels
        self.label_names = label_names or self.DEFAULT_LABELS[:num_labels]
        self.label2id = {label: i for i, label in enumerate(self.label_names)}
        self.id2label = {i: label for i, label in enumerate(self.label_names)}

    def _create_model(
        self, model_name: str, model_config: AutoConfig
    ) -> PreTrainedModel:
        """Create token classification model for operations.

        Args:
            model_name: HuggingFace model name
            model_config: Model configuration

        Returns:
            Token classification model
        """
        model_config.num_labels = self.num_labels
        model_config.id2label = self.id2label
        model_config.label2id = self.label2id
        return AutoModelForTokenClassification.from_pretrained(
            model_name,
            config=model_config,
            cache_dir=self.config.cache_dir,
        )

    def predict(
        self,
        texts: Union[str, list[str]],
        aggregate_operations: bool = True,
    ) -> Union[list[Entity], list[list[Entity]]]:
        """Identify operation tokens in text(s).

        Args:
            texts: Input text(s)
            aggregate_operations: Whether to aggregate BIO tags

        Returns:
            List of recognized operations
        """
        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]

        # Encode inputs
        encoded = self.tokenizer(
            texts,
            max_length=self.config.max_sequence_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
            return_offsets_mapping=True,
        )

        offset_mapping = encoded.pop("offset_mapping")
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        # Forward pass
        with torch.no_grad():
            outputs = self.model(**encoded)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            probabilities = F.softmax(logits, dim=-1)

        # Process results using same logic as MaterialEntityRecognizer
        all_operations = []
        for i in range(len(texts)):
            text = texts[i]
            preds = predictions[i].cpu().numpy()
            probs = probabilities[i].cpu().numpy()
            offsets = offset_mapping[i].numpy()

            if aggregate_operations:
                operations = self._aggregate_operations(text, preds, probs, offsets)
            else:
                operations = self._get_token_predictions(text, preds, probs, offsets)

            all_operations.append(operations)

        return all_operations[0] if single_input else all_operations

    def _aggregate_operations(
        self,
        text: str,
        predictions: Any,
        probabilities: Any,
        offsets: Any,
    ) -> list[Entity]:
        """Aggregate BIO tags into operation spans."""
        operations = []
        current_op = None

        for pred, probs, (start, end) in zip(predictions, probabilities, offsets):
            if start == end == 0:
                continue

            label = self.id2label[pred]
            confidence = float(probs[pred])

            if label.startswith("B-"):
                if current_op is not None:
                    operations.append(current_op)

                op_type = label[2:]
                current_op = Entity(
                    text=text[start:end],
                    label=op_type,
                    start=int(start),
                    end=int(end),
                    confidence=confidence,
                )

            elif label.startswith("I-") and current_op is not None:
                op_type = label[2:]
                if op_type == current_op.label:
                    current_op.text = text[current_op.start : end]
                    current_op.end = int(end)
                    current_op.confidence = min(current_op.confidence, confidence)

            else:
                if current_op is not None:
                    operations.append(current_op)
                    current_op = None

        if current_op is not None:
            operations.append(current_op)

        return operations

    def _get_token_predictions(
        self,
        text: str,
        predictions: Any,
        probabilities: Any,
        offsets: Any,
    ) -> list[dict[str, Any]]:
        """Get per-token predictions."""
        token_preds = []
        for pred, probs, (start, end) in zip(predictions, probabilities, offsets):
            if start == end == 0:
                continue

            token_preds.append({
                "text": text[start:end],
                "label": self.id2label[pred],
                "start": int(start),
                "end": int(end),
                "confidence": float(probs[pred]),
            })

        return token_preds
