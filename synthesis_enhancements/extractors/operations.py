"""Enhanced operations extraction using transformer models."""

from dataclasses import dataclass, field
from typing import Any, Optional, Union
from concurrent.futures import ThreadPoolExecutor
import logging
import re

from tqdm import tqdm

from synthesis_enhancements.models import (
    TransformerModelFactory,
    OperationClassifier,
    OperationTokenClassifier,
    ModelConfig,
)
from synthesis_enhancements.models.token_classifier import Entity
from synthesis_enhancements.processing import (
    BatchProcessor,
    ParallelProcessor,
    ProcessingPipeline,
)
from synthesis_enhancements.utils.config import ExtractorConfig, ProcessingConfig
from synthesis_enhancements.utils.constants import OperationType

logger = logging.getLogger(__name__)


@dataclass
class OperationSpan:
    """A synthesis operation extracted from text.

    Attributes:
        text: The operation text
        operation_type: Type of operation
        start: Start character position
        end: End character position
        confidence: Extraction confidence
        conditions: Extracted conditions (temperature, time, etc.)
        tokens: Constituent tokens
    """
    text: str
    operation_type: str
    start: int
    end: int
    confidence: float
    conditions: dict[str, Any] = field(default_factory=dict)
    tokens: list[str] = field(default_factory=list)


@dataclass
class ExtractionResult:
    """Result of operations extraction.

    Attributes:
        paragraph: Original paragraph text
        operations: Extracted operations
        is_synthesis: Whether paragraph describes synthesis
        synthesis_type: Type of synthesis (if applicable)
        conditions: Global extraction conditions
        metadata: Additional metadata
    """
    paragraph: str
    operations: list[OperationSpan]
    is_synthesis: bool = True
    synthesis_type: Optional[str] = None
    conditions: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class EnhancedOperationsExtractor:
    """Enhanced operations extractor using transformer models.

    Extracts synthesis operations (heating, mixing, drying, etc.)
    from scientific paragraphs using state-of-the-art transformer
    models with parallel processing support.

    Example:
        >>> extractor = EnhancedOperationsExtractor(
        ...     model_name="matbert",
        ...     num_workers=4
        ... )
        >>> extractor.load()
        >>>
        >>> results = extractor.extract([
        ...     "The mixture was heated at 500°C for 2 hours.",
        ...     "After cooling, the sample was ground and sintered."
        ... ])
        >>> for result in results:
        ...     for op in result.operations:
        ...         print(f"{op.operation_type}: {op.text}")

    Attributes:
        config: Extractor configuration
        classifier: Operation classifier model
        token_classifier: Token-level operation classifier
    """

    # Common temperature patterns
    TEMP_PATTERN = re.compile(
        r"(\d+(?:\.\d+)?)\s*°?\s*([CFK]|celsius|fahrenheit|kelvin)",
        re.IGNORECASE,
    )

    # Common time patterns
    TIME_PATTERN = re.compile(
        r"(\d+(?:\.\d+)?)\s*(h(?:our)?s?|min(?:ute)?s?|s(?:ec(?:ond)?)?s?|d(?:ay)?s?)",
        re.IGNORECASE,
    )

    # Common pressure patterns
    PRESSURE_PATTERN = re.compile(
        r"(\d+(?:\.\d+)?)\s*(MPa|GPa|kPa|Pa|atm|bar|psi)",
        re.IGNORECASE,
    )

    def __init__(
        self,
        model_name: str = "matbert",
        num_workers: int = 4,
        batch_size: int = 32,
        config: Optional[ExtractorConfig] = None,
    ) -> None:
        """Initialize the operations extractor.

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

        self._classifier: Optional[OperationClassifier] = None
        self._token_classifier: Optional[OperationTokenClassifier] = None
        self._is_loaded = False

    @property
    def classifier(self) -> OperationClassifier:
        """Get the operation classifier."""
        if self._classifier is None:
            raise RuntimeError("Extractor not loaded. Call load() first.")
        return self._classifier

    @property
    def token_classifier(self) -> OperationTokenClassifier:
        """Get the token classifier."""
        if self._token_classifier is None:
            raise RuntimeError("Extractor not loaded. Call load() first.")
        return self._token_classifier

    def load(self) -> "EnhancedOperationsExtractor":
        """Load the models.

        Returns:
            Self for method chaining
        """
        if self._is_loaded:
            logger.warning("Extractor already loaded")
            return self

        logger.info("Loading operation extraction models...")

        # Create and load sequence classifier
        self._classifier = TransformerModelFactory.create_classifier(
            task="operation",
            model_name=self.config.model_config.model_name,
            num_labels=len(OperationType),
        )
        self._classifier.load()

        # Create and load token classifier
        self._token_classifier = TransformerModelFactory.create_token_classifier(
            task="operation_token",
            model_name=self.config.model_config.model_name,
        )
        self._token_classifier.load()

        self._is_loaded = True
        logger.info("Operation extraction models loaded")
        return self

    def extract(
        self,
        paragraphs: Union[str, list[str]],
        extract_conditions: bool = True,
        use_parallel: bool = True,
    ) -> Union[ExtractionResult, list[ExtractionResult]]:
        """Extract operations from paragraph(s).

        Args:
            paragraphs: Input paragraph(s)
            extract_conditions: Whether to extract conditions
            use_parallel: Whether to use parallel processing

        Returns:
            Extraction result(s)
        """
        single_input = isinstance(paragraphs, str)
        if single_input:
            paragraphs = [paragraphs]

        if use_parallel and len(paragraphs) > 1:
            results = self._extract_parallel(paragraphs, extract_conditions)
        else:
            results = self._extract_sequential(paragraphs, extract_conditions)

        return results[0] if single_input else results

    def _extract_sequential(
        self,
        paragraphs: list[str],
        extract_conditions: bool,
    ) -> list[ExtractionResult]:
        """Extract operations sequentially."""
        results = []
        for para in tqdm(paragraphs, desc="Extracting operations"):
            results.append(self._extract_single(para, extract_conditions))
        return results

    def _extract_parallel(
        self,
        paragraphs: list[str],
        extract_conditions: bool,
    ) -> list[ExtractionResult]:
        """Extract operations in parallel."""

        def process_fn(para: str) -> ExtractionResult:
            return self._extract_single(para, extract_conditions)

        processor = ParallelProcessor(
            process_fn=process_fn,
            config=self.config.processing_config,
            name="OperationsExtraction",
        )

        results = processor.process(paragraphs)
        return [r.output for r in results if r.success]

    def _extract_single(
        self,
        paragraph: str,
        extract_conditions: bool,
    ) -> ExtractionResult:
        """Extract operations from a single paragraph."""
        # Split into sentences for better granularity
        sentences = self._split_sentences(paragraph)

        all_operations = []
        char_offset = 0

        for sentence in sentences:
            # Get token-level predictions
            token_ops = self.token_classifier.predict(sentence)

            # Convert to OperationSpan objects
            for op in token_ops:
                operation = OperationSpan(
                    text=op.text,
                    operation_type=op.label,
                    start=char_offset + op.start,
                    end=char_offset + op.end,
                    confidence=op.confidence,
                )

                # Extract conditions if requested
                if extract_conditions:
                    operation.conditions = self._extract_conditions(
                        paragraph, operation.start, operation.end
                    )

                all_operations.append(operation)

            char_offset += len(sentence) + 1  # +1 for space/newline

        # Refine operations with rule-based corrections
        if self.config.use_rule_refinement:
            all_operations = self._refine_operations(all_operations, paragraph)

        return ExtractionResult(
            paragraph=paragraph,
            operations=all_operations,
            is_synthesis=len(all_operations) > 0,
        )

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]

    def _extract_conditions(
        self,
        text: str,
        start: int,
        end: int,
    ) -> dict[str, Any]:
        """Extract conditions near an operation.

        Args:
            text: Full text
            start: Operation start position
            end: Operation end position

        Returns:
            Dictionary of extracted conditions
        """
        # Look in a window around the operation
        window_size = 100
        window_start = max(0, start - window_size)
        window_end = min(len(text), end + window_size)
        window = text[window_start:window_end]

        conditions = {}

        # Extract temperature
        temp_match = self.TEMP_PATTERN.search(window)
        if temp_match:
            conditions["temperature"] = {
                "value": float(temp_match.group(1)),
                "unit": temp_match.group(2).upper()[0],  # Normalize to C/F/K
            }

        # Extract time
        time_match = self.TIME_PATTERN.search(window)
        if time_match:
            value = float(time_match.group(1))
            unit = time_match.group(2).lower()

            # Normalize units
            if unit.startswith("h"):
                unit = "hours"
            elif unit.startswith("min"):
                unit = "minutes"
            elif unit.startswith("s"):
                unit = "seconds"
            elif unit.startswith("d"):
                unit = "days"

            conditions["time"] = {"value": value, "unit": unit}

        # Extract pressure
        pressure_match = self.PRESSURE_PATTERN.search(window)
        if pressure_match:
            conditions["pressure"] = {
                "value": float(pressure_match.group(1)),
                "unit": pressure_match.group(2),
            }

        return conditions

    def _refine_operations(
        self,
        operations: list[OperationSpan],
        text: str,
    ) -> list[OperationSpan]:
        """Apply rule-based refinements to extracted operations.

        Args:
            operations: Initial operations
            text: Original text

        Returns:
            Refined operations
        """
        refined = []

        for op in operations:
            # Skip low-confidence predictions
            if op.confidence < self.config.confidence_threshold:
                continue

            # Apply operation-specific rules
            if op.operation_type == "Heating":
                # Check for calcination/sintering keywords
                op_text_lower = op.text.lower()
                if "calcin" in op_text_lower:
                    op.operation_type = "Calcining"
                elif "sinter" in op_text_lower:
                    op.operation_type = "Sintering"
                elif "anneal" in op_text_lower:
                    op.operation_type = "Annealing"

            elif op.operation_type == "Cooling":
                # Check for quenching
                if "quench" in op.text.lower():
                    op.operation_type = "Quenching"

            refined.append(op)

        return refined

    def extract_with_pipeline(
        self,
        paragraphs: list[str],
    ) -> list[ExtractionResult]:
        """Extract operations using a processing pipeline.

        Uses a multi-stage pipeline for efficient processing:
        1. Sentence splitting (parallel)
        2. Token classification (batch)
        3. Condition extraction (parallel)
        4. Refinement (sequential)

        Args:
            paragraphs: Input paragraphs

        Returns:
            Extraction results
        """
        pipeline = ProcessingPipeline(show_progress=True)

        # Stage 1: Sentence splitting
        pipeline.add_parallel_stage(
            name="sentence_split",
            process_fn=self._split_sentences,
            num_workers=self.config.processing_config.num_workers,
        )

        # Run pipeline
        result = pipeline.run(paragraphs)

        # Process with models (batched)
        all_results = []
        for para_idx, sentences in enumerate(result.outputs):
            para_operations = []

            # Batch predict on sentences
            if sentences:
                all_token_ops = self.token_classifier.predict(sentences)

                char_offset = 0
                for sent_idx, token_ops in enumerate(all_token_ops):
                    for op in token_ops:
                        operation = OperationSpan(
                            text=op.text,
                            operation_type=op.label,
                            start=char_offset + op.start,
                            end=char_offset + op.end,
                            confidence=op.confidence,
                        )
                        para_operations.append(operation)

                    char_offset += len(sentences[sent_idx]) + 1

            all_results.append(
                ExtractionResult(
                    paragraph=paragraphs[para_idx],
                    operations=para_operations,
                    is_synthesis=len(para_operations) > 0,
                )
            )

        return all_results

    def batch_extract(
        self,
        paragraphs: list[str],
        batch_size: Optional[int] = None,
    ) -> list[ExtractionResult]:
        """Extract operations in batches with multithreading.

        Args:
            paragraphs: Input paragraphs
            batch_size: Batch size (uses config default if None)

        Returns:
            Extraction results
        """
        batch_size = batch_size or self.config.processing_config.batch_size

        def batch_fn(batch: list[str]) -> list[ExtractionResult]:
            return [self._extract_single(p, True) for p in batch]

        processor = BatchProcessor(
            process_fn=batch_fn,
            config=self.config.processing_config,
            name="BatchOperationsExtraction",
        )

        return processor.process_with_threading(paragraphs, batch_size=batch_size)

    def get_operation_summary(
        self,
        results: list[ExtractionResult],
    ) -> dict[str, Any]:
        """Get summary statistics of extracted operations.

        Args:
            results: Extraction results

        Returns:
            Summary statistics
        """
        operation_counts = {}
        total_operations = 0
        paragraphs_with_operations = 0

        for result in results:
            if result.operations:
                paragraphs_with_operations += 1

            for op in result.operations:
                total_operations += 1
                op_type = op.operation_type
                operation_counts[op_type] = operation_counts.get(op_type, 0) + 1

        return {
            "total_paragraphs": len(results),
            "paragraphs_with_operations": paragraphs_with_operations,
            "total_operations": total_operations,
            "operation_counts": operation_counts,
            "avg_operations_per_paragraph": (
                total_operations / len(results) if results else 0
            ),
        }
