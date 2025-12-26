#!/usr/bin/env python3
"""Basic usage examples for synthesis enhancements package."""

from synthesis_enhancements import (
    EnhancedOperationsExtractor,
    EnhancedMaterialEntityRecognizer,
    EnhancedMaterialParser,
)
from synthesis_enhancements.processing import (
    ParallelProcessor,
    BatchProcessor,
    ProcessingPipeline,
)
from synthesis_enhancements.utils.config import ProcessingConfig


def example_operations_extraction():
    """Example: Extract synthesis operations from paragraphs."""
    print("\n" + "="*60)
    print("Operations Extraction Example")
    print("="*60)

    # Sample paragraphs
    paragraphs = [
        "The precursor powders were mixed and heated at 900°C for 12 hours in air.",
        "After cooling to room temperature, the product was ground and calcined at 1100°C.",
        "The mixture was ball-milled for 24 hours, then dried at 80°C overnight.",
    ]

    # Initialize extractor
    extractor = EnhancedOperationsExtractor(
        model_name="matbert",
        num_workers=4,
        batch_size=32,
    )

    # Note: In practice, call extractor.load() to load models
    # extractor.load()

    print("\nSample paragraphs:")
    for i, para in enumerate(paragraphs):
        print(f"  {i+1}. {para}")

    print("\nExtracted operations would include:")
    print("  - Mixing, Heating (900°C, 12h)")
    print("  - Cooling, Grinding, Calcining (1100°C)")
    print("  - Ball-milling (24h), Drying (80°C)")


def example_material_recognition():
    """Example: Recognize material entities and their roles."""
    print("\n" + "="*60)
    print("Material Entity Recognition Example")
    print("="*60)

    text = "LiCoO2 cathode material was synthesized from Li2CO3 and Co3O4 precursors."

    # Initialize recognizer
    recognizer = EnhancedMaterialEntityRecognizer(
        model_name="matbert",
        num_workers=4,
    )

    # Note: In practice, call recognizer.load() to load models
    # recognizer.load()

    print(f"\nInput text: {text}")
    print("\nExpected entities:")
    print("  Targets: LiCoO2")
    print("  Precursors: Li2CO3, Co3O4")


def example_material_parsing():
    """Example: Parse chemical formulas."""
    print("\n" + "="*60)
    print("Material Parsing Example")
    print("="*60)

    materials = [
        "LiCoO2",
        "Li0.5Na0.5MnO2",
        "La1-xSrxMnO3 (x=0.3)",
        "Li2CO3:Co3O4 (1:1 molar)",
    ]

    # Initialize parser
    parser = EnhancedMaterialParser(num_workers=4)
    parser.load(load_embeddings=False)

    print("\nParsing materials:")
    for mat in materials:
        result = parser.parse(mat)
        print(f"\n  Input: {mat}")
        print(f"  Formula: {result.formula}")
        if result.composition:
            comp_str = ", ".join(
                f"{c.element}:{c.amount}" for c in result.composition
            )
            print(f"  Composition: {comp_str}")
        if result.is_mixture:
            print(f"  Type: Mixture with {len(result.components)} components")
        if result.phase:
            print(f"  Phase: {result.phase}")


def example_parallel_processing():
    """Example: Parallel processing infrastructure."""
    print("\n" + "="*60)
    print("Parallel Processing Example")
    print("="*60)

    # Sample data
    data = list(range(100))

    # Define processing function
    def process_item(x):
        return x ** 2

    # Configure processing
    config = ProcessingConfig(
        num_workers=4,
        batch_size=32,
        show_progress=True,
        use_multiprocessing=False,  # Use threading
    )

    # Method 1: ParallelProcessor for item-level parallelism
    print("\n1. ParallelProcessor (item-level parallelism):")
    processor = ParallelProcessor(
        process_fn=process_item,
        config=config,
        name="SquareProcessor",
    )

    results = processor.process(data[:10], show_progress=False)
    print(f"  Input: {data[:10]}")
    print(f"  Output: {[r.output for r in results]}")

    # Method 2: BatchProcessor for batch-level processing
    print("\n2. BatchProcessor (batch-level processing):")

    def batch_process(batch):
        return [x ** 2 for x in batch]

    batch_processor = BatchProcessor(
        process_fn=batch_process,
        config=config,
    )

    results = batch_processor.process(data[:10], show_progress=False)
    print(f"  Input: {data[:10]}")
    print(f"  Output: {results}")

    # Method 3: Processing pipeline
    print("\n3. ProcessingPipeline (chained stages):")
    pipeline = ProcessingPipeline(show_progress=False)
    pipeline.add_stage("double", lambda x: x * 2)
    pipeline.add_stage("add_ten", lambda x: x + 10)
    pipeline.add_stage("square", lambda x: x ** 2)

    result = pipeline.run([1, 2, 3])
    print(f"  Input: [1, 2, 3]")
    print(f"  Stages: double -> add_ten -> square")
    print(f"  Output: {result.outputs}")
    print(f"  Formula: ((x * 2) + 10) ** 2")


def example_batch_with_threading():
    """Example: Batch processing with multithreading."""
    print("\n" + "="*60)
    print("Batch Processing with Multithreading Example")
    print("="*60)

    # Simulate ML model inference
    def model_predict(batch):
        """Simulate batch prediction."""
        import time
        time.sleep(0.01)  # Simulate inference time
        return [f"pred_{x}" for x in batch]

    config = ProcessingConfig(
        batch_size=8,
        num_workers=4,
    )

    processor = BatchProcessor(
        process_fn=model_predict,
        config=config,
        name="ModelInference",
    )

    # Process with threading
    data = list(range(32))
    results = processor.process_with_threading(
        data,
        batch_size=8,
        num_threads=4,
        show_progress=False,
    )

    print(f"\n  Processed {len(data)} items in {len(data)//8} batches")
    print(f"  Using 4 parallel threads")
    print(f"  Sample results: {results[:5]}...")


def main():
    """Run all examples."""
    print("="*60)
    print("Synthesis Enhancements - Usage Examples")
    print("="*60)

    example_operations_extraction()
    example_material_recognition()
    example_material_parsing()
    example_parallel_processing()
    example_batch_with_threading()

    print("\n" + "="*60)
    print("Examples completed!")
    print("="*60)


if __name__ == "__main__":
    main()
