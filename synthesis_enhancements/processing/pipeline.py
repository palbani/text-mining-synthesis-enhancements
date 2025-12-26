"""Processing pipeline for chaining multiple processors."""

from dataclasses import dataclass, field
from typing import Any, Callable, Generic, Optional, TypeVar, Union
from concurrent.futures import ThreadPoolExecutor
import logging
import time

from tqdm import tqdm

from synthesis_enhancements.processing.parallel import ParallelProcessor
from synthesis_enhancements.processing.batch import BatchProcessor
from synthesis_enhancements.utils.config import ProcessingConfig

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


@dataclass
class PipelineStage:
    """A stage in the processing pipeline.

    Attributes:
        name: Stage name
        process_fn: Processing function
        parallel: Whether to use parallel processing
        batch: Whether to use batch processing
        config: Stage-specific configuration
    """
    name: str
    process_fn: Callable
    parallel: bool = False
    batch: bool = False
    batch_size: int = 32
    num_workers: int = 4
    config: Optional[dict[str, Any]] = None

    def __post_init__(self) -> None:
        if self.config is None:
            self.config = {}


@dataclass
class PipelineResult:
    """Result of pipeline execution.

    Attributes:
        outputs: Final outputs
        stage_outputs: Outputs from each stage
        stage_times: Processing time for each stage
        total_time: Total pipeline time
        errors: Errors from each stage
    """
    outputs: list[Any]
    stage_outputs: dict[str, list[Any]] = field(default_factory=dict)
    stage_times: dict[str, float] = field(default_factory=dict)
    total_time: float = 0.0
    errors: dict[str, list[str]] = field(default_factory=dict)


class ProcessingPipeline:
    """Pipeline for chaining multiple processing stages.

    Allows building complex processing workflows by chaining
    together multiple processing stages, each with its own
    configuration for parallel/batch processing.

    Example:
        >>> pipeline = ProcessingPipeline()
        >>> pipeline.add_stage("tokenize", tokenize_fn, parallel=True)
        >>> pipeline.add_stage("classify", classify_fn, batch=True, batch_size=32)
        >>> pipeline.add_stage("postprocess", postprocess_fn)
        >>>
        >>> result = pipeline.run(input_texts)
        >>> print(result.outputs)

    Attributes:
        stages: List of pipeline stages
        keep_intermediate: Whether to keep intermediate outputs
    """

    def __init__(
        self,
        keep_intermediate: bool = False,
        show_progress: bool = True,
    ) -> None:
        """Initialize the pipeline.

        Args:
            keep_intermediate: Whether to keep outputs from each stage
            show_progress: Whether to show progress bars
        """
        self.stages: list[PipelineStage] = []
        self.keep_intermediate = keep_intermediate
        self.show_progress = show_progress

    def add_stage(
        self,
        name: str,
        process_fn: Callable,
        parallel: bool = False,
        batch: bool = False,
        batch_size: int = 32,
        num_workers: int = 4,
        **config: Any,
    ) -> "ProcessingPipeline":
        """Add a stage to the pipeline.

        Args:
            name: Stage name
            process_fn: Processing function
            parallel: Whether to use parallel processing
            batch: Whether to use batch processing
            batch_size: Batch size (if batch=True)
            num_workers: Number of workers (if parallel=True)
            **config: Additional configuration

        Returns:
            Self for method chaining
        """
        stage = PipelineStage(
            name=name,
            process_fn=process_fn,
            parallel=parallel,
            batch=batch,
            batch_size=batch_size,
            num_workers=num_workers,
            config=config,
        )
        self.stages.append(stage)
        return self

    def add_parallel_stage(
        self,
        name: str,
        process_fn: Callable,
        num_workers: int = 4,
        **config: Any,
    ) -> "ProcessingPipeline":
        """Add a parallel processing stage.

        Args:
            name: Stage name
            process_fn: Processing function (applied to each item)
            num_workers: Number of parallel workers
            **config: Additional configuration

        Returns:
            Self for method chaining
        """
        return self.add_stage(
            name=name,
            process_fn=process_fn,
            parallel=True,
            num_workers=num_workers,
            **config,
        )

    def add_batch_stage(
        self,
        name: str,
        process_fn: Callable,
        batch_size: int = 32,
        num_workers: int = 4,
        **config: Any,
    ) -> "ProcessingPipeline":
        """Add a batch processing stage.

        Args:
            name: Stage name
            process_fn: Processing function (applied to batches)
            batch_size: Batch size
            num_workers: Number of workers for parallel batching
            **config: Additional configuration

        Returns:
            Self for method chaining
        """
        return self.add_stage(
            name=name,
            process_fn=process_fn,
            batch=True,
            batch_size=batch_size,
            num_workers=num_workers,
            **config,
        )

    def run(
        self,
        inputs: list[Any],
        start_stage: Optional[str] = None,
        end_stage: Optional[str] = None,
    ) -> PipelineResult:
        """Run the pipeline on inputs.

        Args:
            inputs: List of inputs
            start_stage: Optional stage to start from
            end_stage: Optional stage to end at

        Returns:
            Pipeline result with outputs and metadata
        """
        if not self.stages:
            raise ValueError("Pipeline has no stages")

        result = PipelineResult(
            outputs=inputs,
            stage_outputs={},
            stage_times={},
            errors={},
        )

        current_outputs = inputs
        total_start = time.time()

        # Determine which stages to run
        start_idx = 0
        end_idx = len(self.stages)

        if start_stage:
            start_idx = next(
                (i for i, s in enumerate(self.stages) if s.name == start_stage),
                0,
            )
        if end_stage:
            end_idx = next(
                (i + 1 for i, s in enumerate(self.stages) if s.name == end_stage),
                len(self.stages),
            )

        # Run selected stages
        for stage in self.stages[start_idx:end_idx]:
            logger.info(f"Running stage: {stage.name}")
            stage_start = time.time()

            try:
                current_outputs = self._run_stage(stage, current_outputs)

                if self.keep_intermediate:
                    result.stage_outputs[stage.name] = current_outputs.copy()

            except Exception as e:
                logger.error(f"Stage {stage.name} failed: {e}")
                if stage.name not in result.errors:
                    result.errors[stage.name] = []
                result.errors[stage.name].append(str(e))

            result.stage_times[stage.name] = time.time() - stage_start

        result.outputs = current_outputs
        result.total_time = time.time() - total_start

        return result

    def _run_stage(
        self,
        stage: PipelineStage,
        inputs: list[Any],
    ) -> list[Any]:
        """Run a single pipeline stage.

        Args:
            stage: Pipeline stage
            inputs: Stage inputs

        Returns:
            Stage outputs
        """
        if stage.parallel:
            return self._run_parallel_stage(stage, inputs)
        elif stage.batch:
            return self._run_batch_stage(stage, inputs)
        else:
            return self._run_sequential_stage(stage, inputs)

    def _run_sequential_stage(
        self,
        stage: PipelineStage,
        inputs: list[Any],
    ) -> list[Any]:
        """Run stage sequentially."""
        outputs = []
        iterator = inputs

        if self.show_progress:
            iterator = tqdm(inputs, desc=stage.name)

        for item in iterator:
            outputs.append(stage.process_fn(item))

        return outputs

    def _run_parallel_stage(
        self,
        stage: PipelineStage,
        inputs: list[Any],
    ) -> list[Any]:
        """Run stage in parallel."""
        config = ProcessingConfig(
            num_workers=stage.num_workers,
            show_progress=self.show_progress,
            use_multiprocessing=False,  # Use threading by default in pipeline
        )

        processor = ParallelProcessor(
            process_fn=stage.process_fn,
            config=config,
            name=stage.name,
        )

        results = processor.process(inputs, desc=stage.name)
        return [r.output for r in results]

    def _run_batch_stage(
        self,
        stage: PipelineStage,
        inputs: list[Any],
    ) -> list[Any]:
        """Run stage in batches."""
        config = ProcessingConfig(
            batch_size=stage.batch_size,
            num_workers=stage.num_workers,
            show_progress=self.show_progress,
        )

        processor = BatchProcessor(
            process_fn=stage.process_fn,
            config=config,
            name=stage.name,
        )

        return processor.process(inputs)

    def __repr__(self) -> str:
        stages_str = " -> ".join(s.name for s in self.stages)
        return f"ProcessingPipeline([{stages_str}])"


class ParallelPipelines:
    """Run multiple pipelines in parallel.

    Useful for processing different aspects of data simultaneously.

    Example:
        >>> parallel = ParallelPipelines()
        >>> parallel.add_pipeline("entities", entity_pipeline)
        >>> parallel.add_pipeline("operations", ops_pipeline)
        >>> results = parallel.run(inputs)
    """

    def __init__(self, max_workers: int = 4) -> None:
        """Initialize parallel pipelines.

        Args:
            max_workers: Maximum concurrent pipelines
        """
        self.pipelines: dict[str, ProcessingPipeline] = {}
        self.max_workers = max_workers

    def add_pipeline(
        self,
        name: str,
        pipeline: ProcessingPipeline,
    ) -> "ParallelPipelines":
        """Add a pipeline.

        Args:
            name: Pipeline name
            pipeline: Pipeline instance

        Returns:
            Self for method chaining
        """
        self.pipelines[name] = pipeline
        return self

    def run(
        self,
        inputs: list[Any],
    ) -> dict[str, PipelineResult]:
        """Run all pipelines in parallel.

        Args:
            inputs: Inputs for all pipelines

        Returns:
            Dictionary mapping pipeline names to results
        """
        results = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(pipeline.run, inputs): name
                for name, pipeline in self.pipelines.items()
            }

            for future in futures:
                name = futures[future]
                try:
                    results[name] = future.result()
                except Exception as e:
                    logger.error(f"Pipeline {name} failed: {e}")
                    results[name] = PipelineResult(
                        outputs=[],
                        errors={name: [str(e)]},
                    )

        return results
