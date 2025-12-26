"""Parallel processing infrastructure for synthesis text mining."""

from concurrent.futures import (
    ThreadPoolExecutor,
    ProcessPoolExecutor,
    Future,
    as_completed,
)
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Generic,
    Iterator,
    Optional,
    TypeVar,
    Union,
)
from queue import Queue
from threading import Lock
import logging
import time
import multiprocessing as mp

from tqdm import tqdm

from synthesis_enhancements.utils.config import ProcessingConfig

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


@dataclass
class ProcessingResult(Generic[T]):
    """Result of a processing operation.

    Attributes:
        index: Original index in the input batch
        input_data: Original input data
        output: Processing output (if successful)
        error: Error message (if failed)
        processing_time: Time taken for processing in seconds
        success: Whether processing was successful
    """
    index: int
    input_data: Any
    output: Optional[T] = None
    error: Optional[str] = None
    processing_time: float = 0.0
    success: bool = True

    def __post_init__(self) -> None:
        """Set success flag based on error presence."""
        self.success = self.error is None


@dataclass
class ProcessingError(Exception):
    """Exception for processing errors."""
    message: str
    index: int
    original_error: Optional[Exception] = None

    def __str__(self) -> str:
        return f"Processing error at index {self.index}: {self.message}"


class ParallelProcessor(Generic[T, R]):
    """Generic parallel processor for batch operations.

    Supports both multiprocessing and multithreading with configurable
    worker pools and progress tracking.

    Example:
        >>> processor = ParallelProcessor(
        ...     process_fn=lambda x: x.upper(),
        ...     config=ProcessingConfig(num_workers=4)
        ... )
        >>> results = processor.process(["hello", "world"])
        >>> print([r.output for r in results])
        ['HELLO', 'WORLD']

    Attributes:
        process_fn: Function to apply to each input
        config: Processing configuration
        executor: Thread/process pool executor
    """

    def __init__(
        self,
        process_fn: Callable[[T], R],
        config: Optional[ProcessingConfig] = None,
        name: str = "ParallelProcessor",
    ) -> None:
        """Initialize the parallel processor.

        Args:
            process_fn: Function to apply to each input
            config: Processing configuration
            name: Name for logging purposes
        """
        self.process_fn = process_fn
        self.config = config or ProcessingConfig()
        self.name = name

        self._executor: Optional[Union[ThreadPoolExecutor, ProcessPoolExecutor]] = None
        self._lock = Lock()
        self._stats = {
            "total_processed": 0,
            "total_errors": 0,
            "total_time": 0.0,
        }

    def _get_executor(self) -> Union[ThreadPoolExecutor, ProcessPoolExecutor]:
        """Get or create the executor pool."""
        if self._executor is None:
            if self.config.use_multiprocessing:
                self._executor = ProcessPoolExecutor(
                    max_workers=self.config.num_workers,
                    mp_context=mp.get_context("spawn"),
                )
            else:
                self._executor = ThreadPoolExecutor(
                    max_workers=self.config.num_workers,
                    thread_name_prefix=self.name,
                )
        return self._executor

    def process(
        self,
        inputs: list[T],
        show_progress: Optional[bool] = None,
        desc: Optional[str] = None,
    ) -> list[ProcessingResult[R]]:
        """Process inputs in parallel.

        Args:
            inputs: List of inputs to process
            show_progress: Override config setting for progress bar
            desc: Description for progress bar

        Returns:
            List of processing results in original order
        """
        if not inputs:
            return []

        show_progress = show_progress if show_progress is not None else self.config.show_progress
        desc = desc or self.name

        executor = self._get_executor()
        results: dict[int, ProcessingResult[R]] = {}

        # Submit all tasks
        future_to_index: dict[Future, int] = {}
        for i, item in enumerate(inputs):
            future = executor.submit(self._process_single, i, item)
            future_to_index[future] = i

        # Collect results with optional progress bar
        futures = list(future_to_index.keys())
        if show_progress:
            futures = tqdm(
                as_completed(future_to_index),
                total=len(inputs),
                desc=desc,
            )
        else:
            futures = as_completed(future_to_index)

        for future in futures:
            result = future.result()
            results[result.index] = result

            # Update stats
            with self._lock:
                self._stats["total_processed"] += 1
                self._stats["total_time"] += result.processing_time
                if not result.success:
                    self._stats["total_errors"] += 1

        # Return results in original order
        return [results[i] for i in range(len(inputs))]

    def _process_single(self, index: int, item: T) -> ProcessingResult[R]:
        """Process a single item.

        Args:
            index: Item index
            item: Input item

        Returns:
            Processing result
        """
        start_time = time.time()
        try:
            output = self.process_fn(item)
            return ProcessingResult(
                index=index,
                input_data=item,
                output=output,
                processing_time=time.time() - start_time,
            )
        except Exception as e:
            logger.warning(f"Error processing item {index}: {e}")
            return ProcessingResult(
                index=index,
                input_data=item,
                error=str(e),
                processing_time=time.time() - start_time,
                success=False,
            )

    def process_stream(
        self,
        inputs: Iterator[T],
        buffer_size: Optional[int] = None,
    ) -> Iterator[ProcessingResult[R]]:
        """Process inputs as a stream with bounded buffer.

        Args:
            inputs: Iterator of inputs
            buffer_size: Maximum items to buffer (default: max_queue_size)

        Yields:
            Processing results as they complete
        """
        buffer_size = buffer_size or self.config.max_queue_size
        executor = self._get_executor()

        futures: dict[Future, int] = {}
        index = 0

        for item in inputs:
            # Submit new task
            future = executor.submit(self._process_single, index, item)
            futures[future] = index
            index += 1

            # Check if buffer is full
            while len(futures) >= buffer_size:
                # Wait for any task to complete
                done, _ = as_completed(futures, timeout=0.1).__next__
                for future in done:
                    result = future.result()
                    del futures[future]
                    yield result

        # Process remaining futures
        for future in as_completed(futures):
            yield future.result()

    def map(
        self,
        inputs: list[T],
        show_progress: bool = True,
    ) -> list[R]:
        """Process inputs and return outputs only.

        Convenience method that returns outputs directly,
        raising exceptions for any failed items.

        Args:
            inputs: List of inputs
            show_progress: Whether to show progress bar

        Returns:
            List of outputs

        Raises:
            ProcessingError: If any item fails processing
        """
        results = self.process(inputs, show_progress=show_progress)
        outputs = []

        for result in results:
            if not result.success:
                raise ProcessingError(
                    message=result.error or "Unknown error",
                    index=result.index,
                )
            outputs.append(result.output)

        return outputs

    def get_stats(self) -> dict[str, Any]:
        """Get processing statistics.

        Returns:
            Dictionary of statistics
        """
        with self._lock:
            return {
                **self._stats,
                "avg_time_per_item": (
                    self._stats["total_time"] / self._stats["total_processed"]
                    if self._stats["total_processed"] > 0
                    else 0.0
                ),
                "error_rate": (
                    self._stats["total_errors"] / self._stats["total_processed"]
                    if self._stats["total_processed"] > 0
                    else 0.0
                ),
            }

    def reset_stats(self) -> None:
        """Reset processing statistics."""
        with self._lock:
            self._stats = {
                "total_processed": 0,
                "total_errors": 0,
                "total_time": 0.0,
            }

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the executor pool.

        Args:
            wait: Whether to wait for pending tasks to complete
        """
        if self._executor is not None:
            self._executor.shutdown(wait=wait)
            self._executor = None

    def __enter__(self) -> "ParallelProcessor":
        """Context manager entry."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Context manager exit."""
        self.shutdown()


class WorkerPool:
    """Reusable worker pool for parallel processing.

    Manages a pool of workers that can be shared across
    multiple processing operations.

    Example:
        >>> with WorkerPool(num_workers=4) as pool:
        ...     results1 = pool.map(fn1, data1)
        ...     results2 = pool.map(fn2, data2)
    """

    def __init__(
        self,
        num_workers: int = 4,
        use_multiprocessing: bool = True,
    ) -> None:
        """Initialize the worker pool.

        Args:
            num_workers: Number of workers
            use_multiprocessing: Use processes instead of threads
        """
        self.num_workers = num_workers
        self.use_multiprocessing = use_multiprocessing
        self._executor: Optional[Union[ThreadPoolExecutor, ProcessPoolExecutor]] = None

    def _get_executor(self) -> Union[ThreadPoolExecutor, ProcessPoolExecutor]:
        """Get or create the executor."""
        if self._executor is None:
            if self.use_multiprocessing:
                self._executor = ProcessPoolExecutor(max_workers=self.num_workers)
            else:
                self._executor = ThreadPoolExecutor(max_workers=self.num_workers)
        return self._executor

    def map(
        self,
        fn: Callable[[T], R],
        inputs: list[T],
        show_progress: bool = True,
        desc: str = "Processing",
    ) -> list[R]:
        """Map function over inputs in parallel.

        Args:
            fn: Function to apply
            inputs: Input items
            show_progress: Whether to show progress
            desc: Progress bar description

        Returns:
            List of results
        """
        executor = self._get_executor()
        futures = [executor.submit(fn, item) for item in inputs]

        if show_progress:
            iterator = tqdm(as_completed(futures), total=len(futures), desc=desc)
        else:
            iterator = as_completed(futures)

        # Collect results
        results_map = {}
        for i, future in enumerate(iterator):
            try:
                results_map[futures.index(future)] = future.result()
            except Exception as e:
                logger.error(f"Worker error: {e}")
                results_map[futures.index(future)] = None

        return [results_map[i] for i in range(len(inputs))]

    def submit(
        self,
        fn: Callable[..., R],
        *args: Any,
        **kwargs: Any,
    ) -> Future:
        """Submit a single task.

        Args:
            fn: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Future object
        """
        return self._get_executor().submit(fn, *args, **kwargs)

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the pool."""
        if self._executor is not None:
            self._executor.shutdown(wait=wait)
            self._executor = None

    def __enter__(self) -> "WorkerPool":
        return self

    def __exit__(self, *args: Any) -> None:
        self.shutdown()
