"""Enhanced material entity recognition using transformer models."""

from dataclasses import dataclass, field
from typing import Any, Optional, Union
import logging
import re

from tqdm import tqdm

from synthesis_enhancements.models import (
    TransformerModelFactory,
    MaterialEntityRecognizer,
    ModelConfig,
)
from synthesis_enhancements.models.token_classifier import Entity
from synthesis_enhancements.processing import (
    BatchProcessor,
    ParallelProcessor,
)
from synthesis_enhancements.utils.config import ExtractorConfig, ProcessingConfig
from synthesis_enhancements.utils.constants import MaterialEntityType

logger = logging.getLogger(__name__)


@dataclass
class MaterialEntity:
    """A material entity extracted from text.

    Attributes:
        text: The material text (e.g., "LiCoO2")
        role: Role of the material (Target, Precursor, etc.)
        start: Start character position
        end: End character position
        confidence: Extraction confidence
        formula: Parsed chemical formula (if applicable)
        composition: Elemental composition
        properties: Additional properties
    """
    text: str
    role: str
    start: int
    end: int
    confidence: float
    formula: Optional[str] = None
    composition: dict[str, float] = field(default_factory=dict)
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class MaterialExtractionResult:
    """Result of material entity extraction.

    Attributes:
        text: Original text
        entities: Extracted material entities
        targets: Target materials
        precursors: Precursor materials
        other_materials: Other mentioned materials
        metadata: Additional metadata
    """
    text: str
    entities: list[MaterialEntity]
    targets: list[MaterialEntity] = field(default_factory=list)
    precursors: list[MaterialEntity] = field(default_factory=list)
    other_materials: list[MaterialEntity] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Group entities by role."""
        for entity in self.entities:
            if entity.role == "Target":
                self.targets.append(entity)
            elif entity.role == "Precursor":
                self.precursors.append(entity)
            else:
                self.other_materials.append(entity)


class EnhancedMaterialEntityRecognizer:
    """Enhanced material entity recognizer using transformer models.

    Identifies and classifies material mentions in synthesis text,
    distinguishing between targets, precursors, solvents, dopants, etc.

    Example:
        >>> recognizer = EnhancedMaterialEntityRecognizer(
        ...     model_name="matbert",
        ...     num_workers=4
        ... )
        >>> recognizer.load()
        >>>
        >>> result = recognizer.extract(
        ...     "LiCoO2 was synthesized from Li2CO3 and Co3O4"
        ... )
        >>> print(f"Target: {result.targets[0].text}")
        >>> print(f"Precursors: {[p.text for p in result.precursors]}")

    Attributes:
        config: Extractor configuration
        model: Material entity recognition model
    """

    # Chemical formula pattern
    FORMULA_PATTERN = re.compile(
        r"([A-Z][a-z]?)(\d*\.?\d*)",
    )

    # Common material name patterns
    MATERIAL_PATTERNS = [
        re.compile(r"[A-Z][a-z]?\d*(?:[A-Z][a-z]?\d*)+"),  # Chemical formulas
        re.compile(r"\w+(?:oxide|sulfide|nitride|carbide|phosphate|silicate)s?", re.I),
        re.compile(r"\w+\s+(?:oxide|sulfide|nitride|carbide)", re.I),
    ]

    def __init__(
        self,
        model_name: str = "matbert",
        num_workers: int = 4,
        batch_size: int = 32,
        config: Optional[ExtractorConfig] = None,
    ) -> None:
        """Initialize the material entity recognizer.

        Args:
            model_name: Name of the transformer model
            num_workers: Number of parallel workers
            batch_size: Batch size for processing
            config: Extractor configuration
        """
        self.config = config or ExtractorConfig(
            model_config=ModelConfig(model_name=model_name),
            processing_config=ProcessingConfig(
                num_workers=num_workers,
                batch_size=batch_size,
            ),
        )

        self._model: Optional[MaterialEntityRecognizer] = None
        self._is_loaded = False

    @property
    def model(self) -> MaterialEntityRecognizer:
        """Get the entity recognition model."""
        if self._model is None:
            raise RuntimeError("Recognizer not loaded. Call load() first.")
        return self._model

    def load(self) -> "EnhancedMaterialEntityRecognizer":
        """Load the model.

        Returns:
            Self for method chaining
        """
        if self._is_loaded:
            logger.warning("Recognizer already loaded")
            return self

        logger.info("Loading material entity recognition model...")

        self._model = TransformerModelFactory.create_token_classifier(
            task="material_entity",
            model_name=self.config.model_config.model_name,
        )
        self._model.load()

        self._is_loaded = True
        logger.info("Material entity recognition model loaded")
        return self

    def extract(
        self,
        texts: Union[str, list[str]],
        parse_formulas: bool = True,
        use_parallel: bool = True,
    ) -> Union[MaterialExtractionResult, list[MaterialExtractionResult]]:
        """Extract material entities from text(s).

        Args:
            texts: Input text(s)
            parse_formulas: Whether to parse chemical formulas
            use_parallel: Whether to use parallel processing

        Returns:
            Extraction result(s)
        """
        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]

        if use_parallel and len(texts) > 1:
            results = self._extract_parallel(texts, parse_formulas)
        else:
            results = self._extract_sequential(texts, parse_formulas)

        return results[0] if single_input else results

    def _extract_sequential(
        self,
        texts: list[str],
        parse_formulas: bool,
    ) -> list[MaterialExtractionResult]:
        """Extract entities sequentially."""
        results = []
        for text in tqdm(texts, desc="Extracting materials"):
            results.append(self._extract_single(text, parse_formulas))
        return results

    def _extract_parallel(
        self,
        texts: list[str],
        parse_formulas: bool,
    ) -> list[MaterialExtractionResult]:
        """Extract entities in parallel."""

        def process_fn(text: str) -> MaterialExtractionResult:
            return self._extract_single(text, parse_formulas)

        processor = ParallelProcessor(
            process_fn=process_fn,
            config=self.config.processing_config,
            name="MaterialExtraction",
        )

        results = processor.process(texts)
        return [r.output for r in results if r.success]

    def _extract_single(
        self,
        text: str,
        parse_formulas: bool,
    ) -> MaterialExtractionResult:
        """Extract entities from a single text."""
        # Get model predictions
        entities = self.model.predict(text)

        # Convert to MaterialEntity objects
        material_entities = []
        for entity in entities:
            mat_entity = MaterialEntity(
                text=entity.text,
                role=entity.label,
                start=entity.start,
                end=entity.end,
                confidence=entity.confidence,
            )

            # Parse formula if requested
            if parse_formulas:
                composition = self._parse_formula(entity.text)
                if composition:
                    mat_entity.formula = entity.text
                    mat_entity.composition = composition

            material_entities.append(mat_entity)

        return MaterialExtractionResult(
            text=text,
            entities=material_entities,
        )

    def _parse_formula(self, text: str) -> Optional[dict[str, float]]:
        """Parse a chemical formula into elemental composition.

        Args:
            text: Text that might be a chemical formula

        Returns:
            Dictionary mapping elements to their counts, or None
        """
        composition = {}

        # Try to match formula pattern
        matches = self.FORMULA_PATTERN.findall(text)
        if not matches:
            return None

        for element, count in matches:
            if len(element) > 2:  # Not a valid element symbol
                continue

            count = float(count) if count else 1.0
            composition[element] = composition.get(element, 0) + count

        return composition if composition else None

    def batch_extract(
        self,
        texts: list[str],
        batch_size: Optional[int] = None,
    ) -> list[MaterialExtractionResult]:
        """Extract entities in batches with multithreading.

        Args:
            texts: Input texts
            batch_size: Batch size

        Returns:
            Extraction results
        """
        batch_size = batch_size or self.config.processing_config.batch_size

        # Get all model predictions in batches
        all_entities = self.model.predict(texts)

        # Convert to results
        results = []
        for text, entities in zip(texts, all_entities):
            material_entities = [
                MaterialEntity(
                    text=e.text,
                    role=e.label,
                    start=e.start,
                    end=e.end,
                    confidence=e.confidence,
                    composition=self._parse_formula(e.text) or {},
                )
                for e in entities
            ]

            results.append(
                MaterialExtractionResult(text=text, entities=material_entities)
            )

        return results

    def extract_with_context(
        self,
        paragraphs: list[str],
        context_window: int = 2,
    ) -> list[MaterialExtractionResult]:
        """Extract materials considering surrounding context.

        Uses surrounding paragraphs to improve material role assignment.

        Args:
            paragraphs: List of paragraphs
            context_window: Number of surrounding paragraphs to consider

        Returns:
            Extraction results
        """
        results = []

        for i, para in enumerate(paragraphs):
            # Build context
            start_idx = max(0, i - context_window)
            end_idx = min(len(paragraphs), i + context_window + 1)
            context = " ".join(paragraphs[start_idx:end_idx])

            # Extract from current paragraph
            result = self._extract_single(para, parse_formulas=True)

            # Use context to refine role assignments
            result = self._refine_with_context(result, context)
            results.append(result)

        return results

    def _refine_with_context(
        self,
        result: MaterialExtractionResult,
        context: str,
    ) -> MaterialExtractionResult:
        """Refine entity roles using context.

        Args:
            result: Initial extraction result
            context: Surrounding context

        Returns:
            Refined result
        """
        context_lower = context.lower()

        for entity in result.entities:
            entity_text = entity.text.lower()

            # Look for explicit role indicators in context
            if f"{entity_text} was synthesized" in context_lower:
                entity.role = "Target"
            elif f"from {entity_text}" in context_lower:
                entity.role = "Precursor"
            elif f"dissolved in {entity_text}" in context_lower:
                entity.role = "Solvent"
            elif f"doped with {entity_text}" in context_lower:
                entity.role = "Dopant"

        # Re-group entities
        result.targets = []
        result.precursors = []
        result.other_materials = []

        for entity in result.entities:
            if entity.role == "Target":
                result.targets.append(entity)
            elif entity.role == "Precursor":
                result.precursors.append(entity)
            else:
                result.other_materials.append(entity)

        return result

    def find_reaction_participants(
        self,
        text: str,
    ) -> dict[str, list[MaterialEntity]]:
        """Identify reaction participants from text.

        Args:
            text: Input text

        Returns:
            Dictionary mapping roles to material entities
        """
        result = self._extract_single(text, parse_formulas=True)

        return {
            "targets": result.targets,
            "precursors": result.precursors,
            "solvents": [
                e for e in result.entities if e.role == "Solvent"
            ],
            "dopants": [
                e for e in result.entities if e.role == "Dopant"
            ],
            "catalysts": [
                e for e in result.entities if e.role == "Catalyst"
            ],
            "other": result.other_materials,
        }

    def get_extraction_summary(
        self,
        results: list[MaterialExtractionResult],
    ) -> dict[str, Any]:
        """Get summary statistics of extracted materials.

        Args:
            results: Extraction results

        Returns:
            Summary statistics
        """
        role_counts = {}
        unique_materials = set()
        unique_elements = set()
        total_entities = 0

        for result in results:
            for entity in result.entities:
                total_entities += 1
                unique_materials.add(entity.text)

                role = entity.role
                role_counts[role] = role_counts.get(role, 0) + 1

                for element in entity.composition:
                    unique_elements.add(element)

        return {
            "total_texts": len(results),
            "total_entities": total_entities,
            "unique_materials": len(unique_materials),
            "unique_elements": len(unique_elements),
            "role_counts": role_counts,
            "elements": sorted(unique_elements),
            "avg_entities_per_text": (
                total_entities / len(results) if results else 0
            ),
        }
