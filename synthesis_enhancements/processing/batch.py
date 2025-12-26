"""Batch processing with multithreading support."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Generic, Optional, TypeVar, Union
from threading import Lock, Semaphore
from queue import Queue, Empty
import logging
import time

import torch
import numpy as np
from tqdm import tqdm

from synthesis_enhancements.utils.config import ProcessingConfig

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


@dataclass
class BatchResult(Generic[R]):
    """Result of a batch processing operation.

    Attributes:
        batch_index: Index of the batch
        outputs: List of outputs from the batch
        processing_time: Time taken to process the batch
        batch_size: Number of items in the batch
        errors: List of errors that occurred (if any)
    """
    batch_index: int
    outputs: list[R]
    processing_time: float
    batch_size: int
    errors: list[tuple[int, str]] = None

    def __post_init__(self) -> None:
        if self.errors is None:
            self.errors = []


class BatchProcessor(Generic[T, R]):
    """Batch processor with multithreading for ML model inference.

    Optimized for processing large datasets through ML models
    with configurable batch sizes and worker pools.

    Features:
    - Automatic batching of inputs
    - Multi-threaded preprocessing/postprocessing
    - GPU memory management
    - Progress tracking
    - Error handling with partial results

    Example:
        >>> def model_predict(batch):
        ...     return model(batch)
        >>>
        >>> processor = BatchProcessor(
        ...     process_fn=model_predict,
        ...     config=ProcessingConfig(batch_size=32, num_workers=4)
        ... )
        >>> results = processor.process(large_dataset)

    Attributes:
        process_fn: Batch processing function
        config: Processing configuration
        preprocess_fn: Optional preprocessing function
        postprocess_fn: Optional postprocessing function
    """

    def __init__(
        self,
        process_fn: Callable[[list[T]], list[R]],
        config: Optional[ProcessingConfig] = None,
        preprocess_fn: Optional[Callable[[T], Any]] = None,
        postprocess_fn: Optional[Callable[[Any], R]] = None,
        name: str = "BatchProcessor",
    ) -> None:
        """Initialize the batch processor.

        Args:
            process_fn: Function that processes a batch of inputs
            config: Processing configuration
            preprocess_fn: Optional function to preprocess individual items
            postprocess_fn: Optional function to postprocess individual outputs
            name: Name for logging and progress bar
        """
        self.process_fn = process_fn
        self.config = config or ProcessingConfig()
        self.preprocess_fn = preprocess_fn
        self.postprocess_fn = postprocess_fn
        self.name = name

        self._lock = Lock()
        self._stats = {
            "batches_processed": 0,
            "items_processed": 0,
            "total_time": 0.0,
            "errors": 0,
        }

    def process(
        self,
        inputs: list[T],
        batch_size: Optional[int] = None,
        show_progress: Optional[bool] = None,
    ) -> list[R]:
        """Process inputs in batches.

        Args:
            inputs: List of inputs to process
            batch_size: Override default batch size
            show_progress: Override config setting for progress bar

        Returns:
            List of outputs in original order
        """
        if not inputs:
            return []

        batch_size = batch_size or self.config.batch_size
        show_progress = show_progress if show_progress is not None else self.config.show_progress

        # Create batches
        batches = self._create_batches(inputs, batch_size)
        total_batches = len(batches)

        # Process with optional progress bar
        all_outputs: list[R] = []
        iterator = enumerate(batches)

        if show_progress:
            iterator = tqdm(
                iterator,
                total=total_batches,
                desc=f"{self.name} (batch_size={batch_size})",
            )

        for batch_idx, batch in iterator:
            start_time = time.time()

            try:
                # Preprocess if function provided
                if self.preprocess_fn is not None:
                    batch = self._parallel_preprocess(batch)

                # Process batch
                outputs = self.process_fn(batch)

                # Postprocess if function provided
                if self.postprocess_fn is not None:
                    outputs = self._parallel_postprocess(outputs)

                all_outputs.extend(outputs)

                # Update stats
                with self._lock:
                    self._stats["batches_processed"] += 1
                    self._stats["items_processed"] += len(batch)
                    self._stats["total_time"] += time.time() - start_time

            except Exception as e:
                logger.error(f"Error processing batch {batch_idx}: {e}")
                with self._lock:
                    self._stats["errors"] += 1

                # Fill with None for failed batch items
                all_outputs.extend([None] * len(batch))

        return all_outputs

    def process_with_threading(
        self,
        inputs: list[T],
        batch_size: Optional[int] = None,
        num_threads: Optional[int] = None,
        show_progress: Optional[bool] = None,
    ) -> list[R]:
        """Process inputs using multiple threads for batch processing.

        Each thread processes different batches concurrently. This is
        useful when the processing function releases the GIL (e.g., I/O
        bound operations or native code).

        Args:
            inputs: List of inputs to process
            batch_size: Override default batch size
            num_threads: Number of processing threads
            show_progress: Whether to show progress bar

        Returns:
            List of outputs in original order
        """
        if not inputs:
            return []

        batch_size = batch_size or self.config.batch_size
        num_threads = num_threads or self.config.num_workers
        show_progress = show_progress if show_progress is not None else self.config.show_progress

        # Create batches with indices
        batches = self._create_batches(inputs, batch_size)
        batch_indices = list(range(len(batches)))

        # Process batches in parallel
        results: dict[int, list[R]] = {}

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = {
                executor.submit(self._process_batch, batches[i], i): i
                for i in batch_indices
            }

            iterator = as_completed(futures)
            if show_progress:
                iterator = tqdm(
                    iterator,
                    total=len(batches),
                    desc=f"{self.name} (threads={num_threads})",
                )

            for future in iterator:
                batch_idx = futures[future]
                try:
                    batch_result = future.result()
                    results[batch_idx] = batch_result.outputs
                except Exception as e:
                    logger.error(f"Thread error for batch {batch_idx}: {e}")
                    results[batch_idx] = [None] * len(batches[batch_idx])

        # Reconstruct results in order
        all_outputs = []
        for i in range(len(batches)):
            all_outputs.extend(results.get(i, [None] * len(batches[i])))

        return all_outputs

    def _process_batch(
        self,
        batch: list[T],
        batch_index: int,
    ) -> BatchResult[R]:
        """Process a single batch.

        Args:
            batch: Batch of inputs
            batch_index: Index of the batch

        Returns:
            Batch result
        """
        start_time = time.time()
        errors = []

        try:
            # Preprocess
            if self.preprocess_fn is not None:
                processed_batch = []
                for i, item in enumerate(batch):
                    try:
                        processed_batch.append(self.preprocess_fn(item))
                    except Exception as e:
                        errors.append((i, str(e)))
                        processed_batch.append(None)
                batch = [b for b in processed_batch if b is not None]

            # Process
            outputs = self.process_fn(batch)

            # Postprocess
            if self.postprocess_fn is not None:
                outputs = [self.postprocess_fn(o) for o in outputs]

            return BatchResult(
                batch_index=batch_index,
                outputs=outputs,
                processing_time=time.time() - start_time,
                batch_size=len(batch),
                errors=errors,
            )

        except Exception as e:
            return BatchResult(
                batch_index=batch_index,
                outputs=[],
                processing_time=time.time() - start_time,
                batch_size=len(batch),
                errors=[(0, str(e))],
            )

    def _create_batches(self, inputs: list[T], batch_size: int) -> list[list[T]]:
        """Split inputs into batches.

        Args:
            inputs: List of inputs
            batch_size: Size of each batch

        Returns:
            List of batches
        """
        return [
            inputs[i : i + batch_size]
            for i in range(0, len(inputs), batch_size)
        ]

    def _parallel_preprocess(self, batch: list[T]) -> list[Any]:
        """Preprocess batch items in parallel using threads.

        Args:
            batch: Batch of inputs

        Returns:
            Preprocessed batch
        """
        with ThreadPoolExecutor(max_workers=self.config.num_workers) as executor:
            return list(executor.map(self.preprocess_fn, batch))

    def _parallel_postprocess(self, outputs: list[Any]) -> list[R]:
        """Postprocess outputs in parallel using threads.

        Args:
            outputs: Batch of outputs

        Returns:
            Postprocessed outputs
        """
        with ThreadPoolExecutor(max_workers=self.config.num_workers) as executor:
            return list(executor.map(self.postprocess_fn, outputs))

    def get_stats(self) -> dict[str, Any]:
        """Get processing statistics."""
        with self._lock:
            return {
                **self._stats,
                "avg_batch_time": (
                    self._stats["total_time"] / self._stats["batches_processed"]
                    if self._stats["batches_processed"] > 0
                    else 0.0
                ),
                "items_per_second": (
                    self._stats["items_processed"] / self._stats["total_time"]
                    if self._stats["total_time"] > 0
                    else 0.0
                ),
            }

    def reset_stats(self) -> None:
        """Reset processing statistics."""
        with self._lock:
            self._stats = {
                "batches_processed": 0,
                "items_processed": 0,
                "total_time": 0.0,
                "errors": 0,
            }


class GPUBatchProcessor(BatchProcessor[T, R]):
    """Batch processor optimized for GPU inference.

    Adds GPU memory management, automatic batch size adjustment,
    and efficient data transfer.

    Example:
        >>> processor = GPUBatchProcessor(
        ...     process_fn=model.forward,
        ...     device="cuda:0",
        ...     auto_batch_size=True
        ... )
        >>> results = processor.process(large_dataset)
    """

    def __init__(
        self,
        process_fn: Callable[[list[T]], list[R]],
        device: str = "cuda",
        auto_batch_size: bool = True,
        memory_fraction: float = 0.8,
        **kwargs: Any,
    ) -> None:
        """Initialize GPU batch processor.

        Args:
            process_fn: Batch processing function
            device: GPU device (e.g., 'cuda:0')
            auto_batch_size: Automatically adjust batch size based on memory
            memory_fraction: Fraction of GPU memory to use
            **kwargs: Additional arguments for BatchProcessor
        """
        super().__init__(process_fn, **kwargs)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.auto_batch_size = auto_batch_size
        self.memory_fraction = memory_fraction

        if torch.cuda.is_available():
            self._setup_gpu()

    def _setup_gpu(self) -> None:
        """Set up GPU memory management."""
        # Set memory fraction
        torch.cuda.set_per_process_memory_fraction(
            self.memory_fraction,
            device=self.device,
        )
        # Enable memory caching
        torch.cuda.empty_cache()

    def process(
        self,
        inputs: list[T],
        batch_size: Optional[int] = None,
        **kwargs: Any,
    ) -> list[R]:
        """Process inputs with GPU optimization.

        Args:
            inputs: List of inputs
            batch_size: Override batch size
            **kwargs: Additional arguments

        Returns:
            List of outputs
        """
        if self.auto_batch_size and batch_size is None:
            batch_size = self._estimate_batch_size(inputs)
            logger.info(f"Auto-selected batch size: {batch_size}")

        # Clear GPU cache before processing
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return super().process(inputs, batch_size=batch_size, **kwargs)

    def _estimate_batch_size(self, inputs: list[T]) -> int:
        """Estimate optimal batch size based on GPU memory.

        Args:
            inputs: Input samples

        Returns:
            Estimated batch size
        """
        if not torch.cuda.is_available():
            return self.config.batch_size

        # Get available GPU memory
        total_memory = torch.cuda.get_device_properties(self.device).total_memory
        allocated = torch.cuda.memory_allocated(self.device)
        available = (total_memory - allocated) * self.memory_fraction

        # Estimate memory per sample (rough heuristic)
        # This should be refined based on actual model requirements
        sample_size_estimate = 1024 * 1024  # 1MB per sample as default

        estimated_batch = max(1, int(available / sample_size_estimate))
        return min(estimated_batch, self.config.batch_size)


class StreamingBatchProcessor(Generic[T, R]):
    """Streaming batch processor for continuous data streams.

    Processes data as it arrives, maintaining a buffer and
    processing complete batches.

    Example:
        >>> processor = StreamingBatchProcessor(
        ...     process_fn=model_predict,
        ...     batch_size=32
        ... )
        >>> processor.start()
        >>> for item in data_stream:
        ...     processor.add(item)
        >>> results = processor.flush()
    """

    def __init__(
        self,
        process_fn: Callable[[list[T]], list[R]],
        batch_size: int = 32,
        max_buffer_size: int = 1000,
        num_workers: int = 2,
    ) -> None:
        """Initialize streaming processor.

        Args:
            process_fn: Batch processing function
            batch_size: Size of each batch
            max_buffer_size: Maximum items to buffer
            num_workers: Number of processing workers
        """
        self.process_fn = process_fn
        self.batch_size = batch_size
        self.max_buffer_size = max_buffer_size
        self.num_workers = num_workers

        self._buffer: Queue[T] = Queue(maxsize=max_buffer_size)
        self._results: Queue[R] = Queue()
        self._executor: Optional[ThreadPoolExecutor] = None
        self._semaphore = Semaphore(num_workers)
        self._running = False

    def add(self, item: T, timeout: float = 1.0) -> bool:
        """Add an item to the processing buffer.

        Args:
            item: Item to add
            timeout: Timeout for adding to full buffer

        Returns:
            True if item was added, False if buffer full
        """
        try:
            self._buffer.put(item, timeout=timeout)
            self._check_and_process()
            return True
        except Exception:
            return False

    def _check_and_process(self) -> None:
        """Check if batch is ready and submit for processing."""
        if self._buffer.qsize() >= self.batch_size:
            batch = []
            for _ in range(self.batch_size):
                try:
                    batch.append(self._buffer.get_nowait())
                except Empty:
                    break

            if batch and self._executor is not None:
                self._executor.submit(self._process_batch, batch)

    def _process_batch(self, batch: list[T]) -> None:
        """Process a batch and store results."""
        with self._semaphore:
            try:
                outputs = self.process_fn(batch)
                for output in outputs:
                    self._results.put(output)
            except Exception as e:
                logger.error(f"Batch processing error: {e}")

    def get_results(self, timeout: float = 0.1) -> list[R]:
        """Get available results.

        Args:
            timeout: Timeout for getting results

        Returns:
            List of available results
        """
        results = []
        while True:
            try:
                results.append(self._results.get(timeout=timeout))
            except Empty:
                break
        return results

    def flush(self) -> list[R]:
        """Process remaining items and return all results.

        Returns:
            All remaining results
        """
        # Process remaining items in buffer
        remaining = []
        while not self._buffer.empty():
            try:
                remaining.append(self._buffer.get_nowait())
            except Empty:
                break

        if remaining:
            outputs = self.process_fn(remaining)
            for output in outputs:
                self._results.put(output)

        # Collect all results
        return self.get_results(timeout=1.0)

    def start(self) -> None:
        """Start the processing workers."""
        if not self._running:
            self._executor = ThreadPoolExecutor(max_workers=self.num_workers)
            self._running = True

    def stop(self) -> list[R]:
        """Stop processing and return remaining results.

        Returns:
            Remaining results
        """
        results = self.flush()
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
        self._running = False
        return results

    def __enter__(self) -> "StreamingBatchProcessor":
        self.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self.stop()
