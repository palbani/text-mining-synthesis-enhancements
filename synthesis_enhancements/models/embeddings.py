"""Embedding models for synthesis text mining."""

from typing import Any, Optional, Union
import logging

import torch
import numpy as np
from transformers import AutoConfig, AutoModel, PreTrainedModel

from synthesis_enhancements.models.base import (
    SynthesisTransformerModel,
    TransformerModelFactory,
)
from synthesis_enhancements.utils.config import ModelConfig

logger = logging.getLogger(__name__)


@TransformerModelFactory.register("embedding")
class SynthesisEmbeddingModel(SynthesisTransformerModel):
    """Embedding model for synthesis text.

    Generates dense vector representations of synthesis text
    using transformer models, suitable for similarity search,
    clustering, and downstream tasks.

    Example:
        >>> model = SynthesisEmbeddingModel(
        ...     config=ModelConfig(model_name="matbert")
        ... )
        >>> model.load()
        >>> embeddings = model.embed(["LiCoO2", "NaCl", "Fe2O3"])
        >>> print(embeddings.shape)  # (3, 768)
    """

    def __init__(
        self,
        config: ModelConfig,
        pooling_strategy: str = "mean",
    ) -> None:
        """Initialize the embedding model.

        Args:
            config: Model configuration
            pooling_strategy: How to pool token embeddings
                ('mean', 'max', 'cls', 'first_last_avg')
        """
        super().__init__(config)
        self.pooling_strategy = pooling_strategy

    def _create_model(
        self, model_name: str, model_config: AutoConfig
    ) -> PreTrainedModel:
        """Create base transformer model for embeddings.

        Args:
            model_name: HuggingFace model name
            model_config: Model configuration

        Returns:
            Base transformer model
        """
        return AutoModel.from_pretrained(
            model_name,
            config=model_config,
            cache_dir=self.config.cache_dir,
        )

    def predict(
        self,
        texts: Union[str, list[str]],
        **kwargs: Any,
    ) -> np.ndarray:
        """Generate embeddings for input texts.

        Args:
            texts: Input text(s)
            **kwargs: Additional arguments passed to embed()

        Returns:
            Embedding array of shape (n_texts, embedding_dim)
        """
        return self.embed(texts, **kwargs)

    def embed(
        self,
        texts: Union[str, list[str]],
        normalize: bool = True,
        return_tensor: bool = False,
    ) -> Union[np.ndarray, torch.Tensor]:
        """Generate embeddings for input texts.

        Args:
            texts: Input text(s)
            normalize: Whether to L2-normalize embeddings
            return_tensor: Whether to return PyTorch tensor

        Returns:
            Embedding array/tensor of shape (n_texts, embedding_dim)
        """
        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]

        # Encode inputs
        encoded = self.encode(texts)

        # Forward pass
        with torch.no_grad():
            outputs = self.model(**encoded)

            # Get pooled embeddings
            embeddings = self._pool_embeddings(
                outputs.last_hidden_state,
                encoded.get("attention_mask"),
            )

            # Normalize if requested
            if normalize:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)

        if return_tensor:
            return embeddings[0] if single_input else embeddings

        embeddings_np = embeddings.cpu().numpy()
        return embeddings_np[0] if single_input else embeddings_np

    def _pool_embeddings(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Pool token embeddings into sentence embeddings.

        Args:
            hidden_states: Token embeddings (batch, seq_len, hidden_dim)
            attention_mask: Attention mask (batch, seq_len)

        Returns:
            Pooled embeddings (batch, hidden_dim)
        """
        if self.pooling_strategy == "cls":
            return hidden_states[:, 0, :]

        elif self.pooling_strategy == "mean":
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                sum_embeddings = torch.sum(hidden_states * mask, dim=1)
                sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
                return sum_embeddings / sum_mask
            return hidden_states.mean(dim=1)

        elif self.pooling_strategy == "max":
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).expand(hidden_states.size())
                hidden_states = hidden_states.masked_fill(~mask.bool(), float("-inf"))
            return hidden_states.max(dim=1).values

        elif self.pooling_strategy == "first_last_avg":
            # Average of first and last layer (requires output_hidden_states=True)
            first = hidden_states[:, 0, :]
            last = hidden_states[:, -1, :]
            return (first + last) / 2

        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")

    def similarity(
        self,
        text1: Union[str, list[str]],
        text2: Union[str, list[str]],
    ) -> Union[float, np.ndarray]:
        """Compute cosine similarity between texts.

        Args:
            text1: First text(s)
            text2: Second text(s)

        Returns:
            Similarity score(s)
        """
        emb1 = self.embed(text1, normalize=True)
        emb2 = self.embed(text2, normalize=True)

        if emb1.ndim == 1:
            emb1 = emb1.reshape(1, -1)
        if emb2.ndim == 1:
            emb2 = emb2.reshape(1, -1)

        similarity = np.dot(emb1, emb2.T)

        if similarity.shape == (1, 1):
            return float(similarity[0, 0])
        return similarity

    def batch_embed(
        self,
        texts: list[str],
        batch_size: int = 32,
        normalize: bool = True,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Generate embeddings in batches.

        Args:
            texts: List of input texts
            batch_size: Batch size
            normalize: Whether to normalize embeddings
            show_progress: Whether to show progress bar

        Returns:
            Embedding array of shape (n_texts, embedding_dim)
        """
        from tqdm import tqdm

        all_embeddings = []
        iterator = range(0, len(texts), batch_size)

        if show_progress:
            iterator = tqdm(iterator, desc="Embedding")

        for i in iterator:
            batch = texts[i : i + batch_size]
            embeddings = self.embed(batch, normalize=normalize)
            all_embeddings.append(embeddings)

        return np.vstack(all_embeddings)


class SentenceEmbeddingModel:
    """Wrapper for sentence-transformers models.

    Provides an interface compatible with synthesis enhancement
    models while using the efficient sentence-transformers library.

    Example:
        >>> model = SentenceEmbeddingModel("all-MiniLM-L6-v2")
        >>> embeddings = model.embed(["LiCoO2 synthesis", "NaCl preparation"])
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
    ) -> None:
        """Initialize the sentence embedding model.

        Args:
            model_name: Sentence-transformers model name
            device: Device to run on
        """
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device=device)
        self._device = self.model.device

    @property
    def device(self) -> torch.device:
        """Get model device."""
        return self._device

    def embed(
        self,
        texts: Union[str, list[str]],
        normalize: bool = True,
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Generate embeddings for texts.

        Args:
            texts: Input text(s)
            normalize: Whether to normalize embeddings
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bar

        Returns:
            Embedding array
        """
        return self.model.encode(
            texts,
            normalize_embeddings=normalize,
            batch_size=batch_size,
            show_progress_bar=show_progress,
        )

    def similarity(
        self,
        text1: Union[str, list[str]],
        text2: Union[str, list[str]],
    ) -> Union[float, np.ndarray]:
        """Compute similarity between texts."""
        emb1 = self.embed(text1, normalize=True)
        emb2 = self.embed(text2, normalize=True)

        if emb1.ndim == 1:
            emb1 = emb1.reshape(1, -1)
        if emb2.ndim == 1:
            emb2 = emb2.reshape(1, -1)

        similarity = np.dot(emb1, emb2.T)

        if similarity.shape == (1, 1):
            return float(similarity[0, 0])
        return similarity
