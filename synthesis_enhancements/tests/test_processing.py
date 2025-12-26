"""Tests for parallel processing infrastructure."""

import pytest
from unittest.mock import Mock, patch
import time
from concurrent.futures import ThreadPoolExecutor

from synthesis_enhancements.processing.parallel import (
    ParallelProcessor,
    ProcessingResult,
    ProcessingError,
    WorkerPool,
)
from synthesis_enhancements.processing.batch import (
    BatchProcessor,
    BatchResult,
    StreamingBatchProcessor,
)
from synthesis_enhancements.processing.pipeline import (
    ProcessingPipeline,
    PipelineStage,
)
from synthesis_enhancements.utils.config import ProcessingConfig


class TestProcessingResult:
    """Tests for ProcessingResult."""

    def test_successful_result(self):
        """Test successful processing result."""
        result = ProcessingResult(
            index=0,
            input_data="test",
            output="TEST",
            processing_time=0.1,
        )

        assert result.success is True
        assert result.error is None
        assert result.output == "TEST"

    def test_failed_result(self):
        """Test failed processing result."""
        result = ProcessingResult(
            index=1,
            input_data="test",
            error="Something went wrong",
            processing_time=0.05,
        )

        assert result.success is False
        assert result.error == "Something went wrong"
        assert result.output is None


class TestParallelProcessor:
    """Tests for ParallelProcessor."""

    def test_initialization(self):
        """Test processor initialization."""
        processor = ParallelProcessor(
            process_fn=lambda x: x.upper(),
            config=ProcessingConfig(num_workers=4),
        )

        assert processor.config.num_workers == 4

    def test_process_empty_list(self):
        """Test processing empty list."""
        processor = ParallelProcessor(process_fn=lambda x: x)
        results = processor.process([])

        assert results == []

    def test_process_single_item(self):
        """Test processing single item."""
        processor = ParallelProcessor(
            process_fn=lambda x: x.upper(),
            config=ProcessingConfig(num_workers=2, use_multiprocessing=False),
        )

        results = processor.process(["hello"], show_progress=False)

        assert len(results) == 1
        assert results[0].output == "HELLO"
        assert results[0].success is True

    def test_process_multiple_items(self):
        """Test processing multiple items."""
        processor = ParallelProcessor(
            process_fn=lambda x: x * 2,
            config=ProcessingConfig(num_workers=2, use_multiprocessing=False),
        )

        results = processor.process([1, 2, 3], show_progress=False)

        assert len(results) == 3
        assert [r.output for r in results] == [2, 4, 6]

    def test_process_maintains_order(self):
        """Test that results maintain original order."""
        def slow_process(x):
            time.sleep(0.01 * (10 - x))  # Slower for smaller numbers
            return x * 2

        processor = ParallelProcessor(
            process_fn=slow_process,
            config=ProcessingConfig(num_workers=4, use_multiprocessing=False),
        )

        inputs = list(range(10))
        results = processor.process(inputs, show_progress=False)

        # Results should be in original order despite varying processing times
        assert [r.index for r in results] == inputs
        assert [r.output for r in results] == [x * 2 for x in inputs]

    def test_process_handles_errors(self):
        """Test error handling during processing."""
        def failing_fn(x):
            if x == 2:
                raise ValueError("Failed on 2")
            return x * 2

        processor = ParallelProcessor(
            process_fn=failing_fn,
            config=ProcessingConfig(num_workers=2, use_multiprocessing=False),
        )

        results = processor.process([1, 2, 3], show_progress=False)

        assert results[0].success is True
        assert results[0].output == 2

        assert results[1].success is False
        assert "Failed on 2" in results[1].error

        assert results[2].success is True
        assert results[2].output == 6

    def test_map_raises_on_error(self):
        """Test that map raises ProcessingError on failure."""
        def failing_fn(x):
            if x == 2:
                raise ValueError("Error")
            return x

        processor = ParallelProcessor(
            process_fn=failing_fn,
            config=ProcessingConfig(num_workers=2, use_multiprocessing=False),
        )

        with pytest.raises(ProcessingError):
            processor.map([1, 2, 3], show_progress=False)

    def test_statistics(self):
        """Test processing statistics."""
        processor = ParallelProcessor(
            process_fn=lambda x: x,
            config=ProcessingConfig(num_workers=2, use_multiprocessing=False),
        )

        processor.process([1, 2, 3], show_progress=False)
        stats = processor.get_stats()

        assert stats["total_processed"] == 3
        assert stats["total_errors"] == 0
        assert stats["total_time"] > 0

    def test_context_manager(self):
        """Test context manager usage."""
        with ParallelProcessor(
            process_fn=lambda x: x,
            config=ProcessingConfig(use_multiprocessing=False),
        ) as processor:
            results = processor.process([1, 2], show_progress=False)
            assert len(results) == 2


class TestBatchProcessor:
    """Tests for BatchProcessor."""

    def test_batch_creation(self):
        """Test internal batch creation."""
        processor = BatchProcessor(
            process_fn=lambda batch: [x * 2 for x in batch],
            config=ProcessingConfig(batch_size=3),
        )

        batches = processor._create_batches([1, 2, 3, 4, 5], 3)

        assert len(batches) == 2
        assert batches[0] == [1, 2, 3]
        assert batches[1] == [4, 5]

    def test_process_batches(self):
        """Test batch processing."""
        processor = BatchProcessor(
            process_fn=lambda batch: [x.upper() for x in batch],
            config=ProcessingConfig(batch_size=2),
        )

        results = processor.process(["a", "b", "c", "d"], show_progress=False)

        assert results == ["A", "B", "C", "D"]

    def test_batch_statistics(self):
        """Test batch processing statistics."""
        processor = BatchProcessor(
            process_fn=lambda batch: batch,
            config=ProcessingConfig(batch_size=2),
        )

        processor.process([1, 2, 3, 4, 5], show_progress=False)
        stats = processor.get_stats()

        assert stats["batches_processed"] == 3
        assert stats["items_processed"] == 5

    def test_preprocess_function(self):
        """Test preprocessing function."""
        processor = BatchProcessor(
            process_fn=lambda batch: batch,
            preprocess_fn=lambda x: x * 2,
            config=ProcessingConfig(batch_size=2, num_workers=1),
        )

        results = processor.process([1, 2, 3], show_progress=False)
        assert results == [2, 4, 6]

    def test_postprocess_function(self):
        """Test postprocessing function."""
        processor = BatchProcessor(
            process_fn=lambda batch: batch,
            postprocess_fn=lambda x: x + 10,
            config=ProcessingConfig(batch_size=2, num_workers=1),
        )

        results = processor.process([1, 2, 3], show_progress=False)
        assert results == [11, 12, 13]


class TestProcessingPipeline:
    """Tests for ProcessingPipeline."""

    def test_add_stage(self):
        """Test adding stages to pipeline."""
        pipeline = ProcessingPipeline()
        pipeline.add_stage("stage1", lambda x: x)
        pipeline.add_stage("stage2", lambda x: x)

        assert len(pipeline.stages) == 2
        assert pipeline.stages[0].name == "stage1"
        assert pipeline.stages[1].name == "stage2"

    def test_method_chaining(self):
        """Test method chaining for adding stages."""
        pipeline = (
            ProcessingPipeline()
            .add_stage("s1", lambda x: x)
            .add_stage("s2", lambda x: x)
        )

        assert len(pipeline.stages) == 2

    def test_empty_pipeline_raises(self):
        """Test that empty pipeline raises error."""
        pipeline = ProcessingPipeline()

        with pytest.raises(ValueError, match="no stages"):
            pipeline.run([1, 2, 3])

    def test_sequential_stages(self):
        """Test sequential stage execution."""
        pipeline = ProcessingPipeline(show_progress=False)
        pipeline.add_stage("double", lambda x: x * 2)
        pipeline.add_stage("add_one", lambda x: x + 1)

        result = pipeline.run([1, 2, 3])

        assert result.outputs == [3, 5, 7]  # (1*2)+1, (2*2)+1, (3*2)+1

    def test_stage_timing(self):
        """Test that stage timing is recorded."""
        pipeline = ProcessingPipeline(show_progress=False)
        pipeline.add_stage("stage1", lambda x: x)

        result = pipeline.run([1, 2, 3])

        assert "stage1" in result.stage_times
        assert result.stage_times["stage1"] >= 0

    def test_keep_intermediate(self):
        """Test keeping intermediate outputs."""
        pipeline = ProcessingPipeline(keep_intermediate=True, show_progress=False)
        pipeline.add_stage("double", lambda x: x * 2)
        pipeline.add_stage("triple", lambda x: x * 3)

        result = pipeline.run([1, 2])

        assert "double" in result.stage_outputs
        assert result.stage_outputs["double"] == [2, 4]

    def test_repr(self):
        """Test pipeline string representation."""
        pipeline = ProcessingPipeline()
        pipeline.add_stage("a", lambda x: x)
        pipeline.add_stage("b", lambda x: x)

        assert "a -> b" in repr(pipeline)


class TestWorkerPool:
    """Tests for WorkerPool."""

    def test_map_function(self):
        """Test map operation."""
        with WorkerPool(num_workers=2, use_multiprocessing=False) as pool:
            results = pool.map(
                lambda x: x * 2,
                [1, 2, 3],
                show_progress=False,
            )

        assert results == [2, 4, 6]

    def test_submit_task(self):
        """Test submitting individual tasks."""
        with WorkerPool(num_workers=2, use_multiprocessing=False) as pool:
            future = pool.submit(lambda x: x * 2, 5)
            result = future.result()

        assert result == 10
