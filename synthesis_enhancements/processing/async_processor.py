"""Asynchronous processing for synthesis text mining."""

import asyncio
from asyncio import Semaphore as AsyncSemaphore
from dataclasses import dataclass
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Generic,
    Optional,
    TypeVar,
    Union,
)
import logging
import time

from tqdm.asyncio import tqdm as async_tqdm

from synthesis_enhancements.utils.config import ProcessingConfig

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


@dataclass
class AsyncResult(Generic[R]):
    """Result of an async processing operation.

    Attributes:
        index: Original index in the input
        output: Processing output (if successful)
        error: Error message (if failed)
        processing_time: Time taken for processing
        success: Whether processing was successful
    """
    index: int
    output: Optional[R] = None
    error: Optional[str] = None
    processing_time: float = 0.0
    success: bool = True

    def __post_init__(self) -> None:
        self.success = self.error is None


class AsyncProcessor(Generic[T, R]):
    """Asynchronous processor for I/O-bound operations.

    Optimized for operations that involve network calls, file I/O,
    or other async-friendly workloads.

    Example:
        >>> async def fetch_data(url):
        ...     async with aiohttp.ClientSession() as session:
        ...         async with session.get(url) as response:
        ...             return await response.json()
        >>>
        >>> processor = AsyncProcessor(
        ...     process_fn=fetch_data,
        ...     max_concurrent=10
        ... )
        >>> results = await processor.process(urls)

    Attributes:
        process_fn: Async function to apply to each input
        max_concurrent: Maximum concurrent operations
        timeout: Timeout for each operation
    """

    def __init__(
        self,
        process_fn: Callable[[T], Awaitable[R]],
        max_concurrent: int = 10,
        timeout: float = 30.0,
        retry_count: int = 3,
        retry_delay: float = 1.0,
        config: Optional[ProcessingConfig] = None,
    ) -> None:
        """Initialize the async processor.

        Args:
            process_fn: Async function to process each input
            max_concurrent: Maximum concurrent operations
            timeout: Timeout per operation in seconds
            retry_count: Number of retries on failure
            retry_delay: Delay between retries in seconds
            config: Processing configuration
        """
        self.process_fn = process_fn
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self.retry_count = retry_count
        self.retry_delay = retry_delay
        self.config = config or ProcessingConfig()

        self._semaphore: Optional[AsyncSemaphore] = None
        self._stats = {
            "total_processed": 0,
            "total_errors": 0,
            "total_retries": 0,
            "total_time": 0.0,
        }

    async def process(
        self,
        inputs: list[T],
        show_progress: bool = True,
        desc: str = "Processing",
    ) -> list[AsyncResult[R]]:
        """Process inputs asynchronously.

        Args:
            inputs: List of inputs to process
            show_progress: Whether to show progress bar
            desc: Description for progress bar

        Returns:
            List of results in original order
        """
        if not inputs:
            return []

        self._semaphore = AsyncSemaphore(self.max_concurrent)

        # Create tasks for all inputs
        tasks = [
            self._process_with_retry(i, item)
            for i, item in enumerate(inputs)
        ]

        # Process with progress bar
        if show_progress:
            results = []
            for coro in async_tqdm.as_completed(tasks, total=len(tasks), desc=desc):
                result = await coro
                results.append(result)
        else:
            results = await asyncio.gather(*tasks)

        # Sort by index to maintain order
        results.sort(key=lambda r: r.index)
        return results

    async def _process_with_retry(
        self,
        index: int,
        item: T,
    ) -> AsyncResult[R]:
        """Process a single item with retries.

        Args:
            index: Item index
            item: Input item

        Returns:
            Processing result
        """
        last_error = None
        start_time = time.time()

        for attempt in range(self.retry_count + 1):
            try:
                async with self._semaphore:
                    output = await asyncio.wait_for(
                        self.process_fn(item),
                        timeout=self.timeout,
                    )

                self._stats["total_processed"] += 1
                return AsyncResult(
                    index=index,
                    output=output,
                    processing_time=time.time() - start_time,
                )

            except asyncio.TimeoutError:
                last_error = f"Timeout after {self.timeout}s"
                logger.warning(f"Timeout for item {index}, attempt {attempt + 1}")

            except Exception as e:
                last_error = str(e)
                logger.warning(
                    f"Error for item {index}, attempt {attempt + 1}: {e}"
                )

            if attempt < self.retry_count:
                self._stats["total_retries"] += 1
                await asyncio.sleep(self.retry_delay * (attempt + 1))

        self._stats["total_errors"] += 1
        return AsyncResult(
            index=index,
            error=last_error,
            processing_time=time.time() - start_time,
            success=False,
        )

    async def process_stream(
        self,
        inputs: AsyncIterator[T],
        buffer_size: int = 100,
    ) -> AsyncIterator[AsyncResult[R]]:
        """Process an async stream of inputs.

        Args:
            inputs: Async iterator of inputs
            buffer_size: Maximum items to buffer

        Yields:
            Results as they complete
        """
        self._semaphore = AsyncSemaphore(self.max_concurrent)

        pending: set[asyncio.Task] = set()
        index = 0

        async for item in inputs:
            # Create task for this item
            task = asyncio.create_task(
                self._process_with_retry(index, item)
            )
            pending.add(task)
            index += 1

            # If buffer full, wait for some tasks to complete
            while len(pending) >= buffer_size:
                done, pending = await asyncio.wait(
                    pending,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in done:
                    yield task.result()

        # Process remaining tasks
        if pending:
            done, _ = await asyncio.wait(pending)
            for task in done:
                yield task.result()

    async def map(
        self,
        inputs: list[T],
        show_progress: bool = True,
    ) -> list[R]:
        """Process inputs and return outputs only.

        Convenience method that returns outputs directly.
        Failed items are returned as None.

        Args:
            inputs: List of inputs
            show_progress: Whether to show progress bar

        Returns:
            List of outputs (None for failed items)
        """
        results = await self.process(inputs, show_progress=show_progress)
        return [r.output for r in results]

    def get_stats(self) -> dict[str, Any]:
        """Get processing statistics."""
        return {
            **self._stats,
            "success_rate": (
                (self._stats["total_processed"] - self._stats["total_errors"])
                / max(self._stats["total_processed"], 1)
            ),
            "retry_rate": (
                self._stats["total_retries"]
                / max(self._stats["total_processed"], 1)
            ),
        }

    def reset_stats(self) -> None:
        """Reset processing statistics."""
        self._stats = {
            "total_processed": 0,
            "total_errors": 0,
            "total_retries": 0,
            "total_time": 0.0,
        }


class AsyncBatchProcessor(Generic[T, R]):
    """Async processor that batches inputs before processing.

    Combines async processing with batching for efficient
    handling of large datasets.

    Example:
        >>> async def batch_predict(batch):
        ...     return await model.async_forward(batch)
        >>>
        >>> processor = AsyncBatchProcessor(
        ...     process_fn=batch_predict,
        ...     batch_size=32,
        ...     max_concurrent_batches=4
        ... )
        >>> results = await processor.process(large_dataset)
    """

    def __init__(
        self,
        process_fn: Callable[[list[T]], Awaitable[list[R]]],
        batch_size: int = 32,
        max_concurrent_batches: int = 4,
        timeout: float = 60.0,
    ) -> None:
        """Initialize async batch processor.

        Args:
            process_fn: Async function that processes a batch
            batch_size: Number of items per batch
            max_concurrent_batches: Maximum concurrent batch operations
            timeout: Timeout per batch in seconds
        """
        self.process_fn = process_fn
        self.batch_size = batch_size
        self.max_concurrent_batches = max_concurrent_batches
        self.timeout = timeout

    async def process(
        self,
        inputs: list[T],
        show_progress: bool = True,
    ) -> list[R]:
        """Process inputs in async batches.

        Args:
            inputs: List of inputs
            show_progress: Whether to show progress bar

        Returns:
            List of outputs in original order
        """
        if not inputs:
            return []

        # Create batches
        batches = [
            inputs[i : i + self.batch_size]
            for i in range(0, len(inputs), self.batch_size)
        ]

        semaphore = AsyncSemaphore(self.max_concurrent_batches)

        async def process_batch(batch_idx: int, batch: list[T]) -> tuple[int, list[R]]:
            async with semaphore:
                try:
                    outputs = await asyncio.wait_for(
                        self.process_fn(batch),
                        timeout=self.timeout,
                    )
                    return batch_idx, outputs
                except Exception as e:
                    logger.error(f"Batch {batch_idx} failed: {e}")
                    return batch_idx, [None] * len(batch)

        # Create tasks for all batches
        tasks = [
            process_batch(i, batch)
            for i, batch in enumerate(batches)
        ]

        # Process with progress bar
        if show_progress:
            results = []
            for coro in async_tqdm.as_completed(tasks, total=len(tasks), desc="Batches"):
                result = await coro
                results.append(result)
        else:
            results = await asyncio.gather(*tasks)

        # Sort by batch index and flatten
        results.sort(key=lambda x: x[0])
        return [item for _, batch in results for item in batch]


def run_async(coro: Awaitable[T]) -> T:
    """Run an async coroutine synchronously.

    Utility function for using async processors in sync code.

    Args:
        coro: Coroutine to run

    Returns:
        Result of the coroutine
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop.run_until_complete(coro)
