"""Tests for transformer models."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import torch
import numpy as np

from synthesis_enhancements.models.base import (
    SynthesisTransformerModel,
    TransformerModelFactory,
)
from synthesis_enhancements.models.sequence_classifier import (
    OperationClassifier,
    ParagraphClassifier,
)
from synthesis_enhancements.models.token_classifier import (
    MaterialEntityRecognizer,
    Entity,
)
from synthesis_enhancements.models.embeddings import SynthesisEmbeddingModel
from synthesis_enhancements.utils.config import ModelConfig


class TestModelConfig:
    """Tests for ModelConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ModelConfig()
        assert config.model_name == "matbert"
        assert config.max_sequence_length == 512
        assert config.device == "auto"
        assert config.use_fp16 is False

    def test_custom_config(self):
        """Test custom configuration."""
        config = ModelConfig(
            model_name="scibert",
            max_sequence_length=256,
            device="cpu",
        )
        assert config.model_name == "scibert"
        assert config.max_sequence_length == 256
        assert config.device == "cpu"

    def test_invalid_model_name(self):
        """Test that invalid model name raises error."""
        with pytest.raises(ValueError, match="Unknown model"):
            ModelConfig(model_name="invalid_model")

    def test_custom_model_path(self):
        """Test custom model path bypasses validation."""
        config = ModelConfig(
            model_name="custom",
            model_path="/path/to/model",
        )
        assert config.hf_model_name == "/path/to/model"

    def test_hf_model_name_property(self):
        """Test HuggingFace model name resolution."""
        config = ModelConfig(model_name="matbert")
        assert config.hf_model_name == "m3rg-iitd/matscibert"

        config = ModelConfig(model_name="scibert")
        assert config.hf_model_name == "allenai/scibert_scivocab_uncased"


class TestTransformerModelFactory:
    """Tests for TransformerModelFactory."""

    def test_list_available_models(self):
        """Test listing available models."""
        models = TransformerModelFactory.list_available_models()
        assert "matbert" in models
        assert "scibert" in models
        assert "bert-base" in models

    def test_create_classifier(self):
        """Test classifier creation."""
        classifier = TransformerModelFactory.create_classifier(
            task="operation",
            model_name="matbert",
            num_labels=7,
        )
        assert isinstance(classifier, OperationClassifier)
        assert classifier.num_labels == 7

    def test_create_unknown_task(self):
        """Test that unknown task raises error."""
        with pytest.raises(ValueError, match="Unknown task"):
            TransformerModelFactory.create("unknown_task")


class TestOperationClassifier:
    """Tests for OperationClassifier."""

    def test_initialization(self):
        """Test classifier initialization."""
        config = ModelConfig(model_name="matbert")
        classifier = OperationClassifier(config, num_labels=7)

        assert classifier.num_labels == 7
        assert len(classifier.label_names) == 7

    def test_custom_labels(self):
        """Test custom label names."""
        labels = ["Op1", "Op2", "Op3"]
        config = ModelConfig(model_name="matbert")
        classifier = OperationClassifier(
            config,
            num_labels=3,
            label_names=labels,
        )

        assert classifier.label_names == labels

    @patch.object(OperationClassifier, 'load')
    @patch.object(OperationClassifier, 'encode')
    def test_predict_returns_dict(self, mock_encode, mock_load):
        """Test that predict returns proper structure."""
        config = ModelConfig(model_name="matbert")
        classifier = OperationClassifier(config, num_labels=3)

        # Mock the model
        classifier._model = Mock()
        classifier._tokenizer = Mock()
        classifier._is_loaded = True

        # Mock encode output
        mock_encode.return_value = {
            "input_ids": torch.zeros(1, 10),
            "attention_mask": torch.ones(1, 10),
        }

        # Mock model output
        mock_output = Mock()
        mock_output.logits = torch.randn(1, 3)
        classifier._model.return_value = mock_output

        result = classifier.predict("Test text")

        assert "label" in result
        assert "confidence" in result
        assert "label_id" in result


class TestMaterialEntityRecognizer:
    """Tests for MaterialEntityRecognizer."""

    def test_default_labels(self):
        """Test default BIO labels."""
        config = ModelConfig(model_name="matbert")
        recognizer = MaterialEntityRecognizer(config)

        assert "O" in recognizer.label_names
        assert "B-Target" in recognizer.label_names
        assert "I-Target" in recognizer.label_names
        assert "B-Precursor" in recognizer.label_names

    def test_label_mappings(self):
        """Test label to ID mappings."""
        config = ModelConfig(model_name="matbert")
        recognizer = MaterialEntityRecognizer(config)

        assert recognizer.label2id["O"] == 0
        assert recognizer.id2label[0] == "O"


class TestEntity:
    """Tests for Entity dataclass."""

    def test_entity_creation(self):
        """Test entity creation."""
        entity = Entity(
            text="LiCoO2",
            label="Target",
            start=0,
            end=6,
            confidence=0.95,
        )

        assert entity.text == "LiCoO2"
        assert entity.label == "Target"
        assert entity.confidence == 0.95

    def test_entity_to_dict(self):
        """Test entity to dictionary conversion."""
        entity = Entity(
            text="Li2CO3",
            label="Precursor",
            start=10,
            end=16,
            confidence=0.88,
        )

        d = entity.to_dict()
        assert d["text"] == "Li2CO3"
        assert d["label"] == "Precursor"
        assert d["start"] == 10
        assert d["end"] == 16
        assert d["confidence"] == 0.88


class TestSynthesisEmbeddingModel:
    """Tests for SynthesisEmbeddingModel."""

    def test_initialization(self):
        """Test embedding model initialization."""
        config = ModelConfig(model_name="matbert")
        model = SynthesisEmbeddingModel(config, pooling_strategy="mean")

        assert model.pooling_strategy == "mean"

    def test_pooling_strategies(self):
        """Test different pooling strategies are accepted."""
        config = ModelConfig(model_name="matbert")

        for strategy in ["mean", "max", "cls", "first_last_avg"]:
            model = SynthesisEmbeddingModel(config, pooling_strategy=strategy)
            assert model.pooling_strategy == strategy
