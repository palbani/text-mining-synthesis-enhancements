#!/usr/bin/env python3
"""
Standalone test script - Run this directly in Visual Studio or command line.
No model downloads required - tests the core processing and parsing functionality.

Usage:
    python test_standalone.py
"""

import sys


def test_parallel_processor():
    """Test the parallel processing infrastructure."""
    print("\n" + "="*60)
    print("TEST 1: Parallel Processor")
    print("="*60)

    from synthesis_enhancements.processing import ParallelProcessor
    from synthesis_enhancements.utils.config import ProcessingConfig

    # Create processor that squares numbers
    config = ProcessingConfig(num_workers=4, use_multiprocessing=False)
    processor = ParallelProcessor(
        process_fn=lambda x: x ** 2,
        config=config,
        name="SquareProcessor"
    )

    # Process data
    inputs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    results = processor.process(inputs, show_progress=False)
    outputs = [r.output for r in results]

    expected = [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]
    assert outputs == expected, f"Expected {expected}, got {outputs}"

    print(f"  Input:    {inputs}")
    print(f"  Output:   {outputs}")
    print(f"  ✓ PASSED")

    # Show stats
    stats = processor.get_stats()
    print(f"  Stats: {stats['total_processed']} items processed")


def test_batch_processor():
    """Test batch processing with multithreading."""
    print("\n" + "="*60)
    print("TEST 2: Batch Processor with Multithreading")
    print("="*60)

    from synthesis_enhancements.processing import BatchProcessor
    from synthesis_enhancements.utils.config import ProcessingConfig

    # Create batch processor
    def process_batch(batch):
        return [x.upper() for x in batch]

    config = ProcessingConfig(batch_size=3, num_workers=2)
    processor = BatchProcessor(
        process_fn=process_batch,
        config=config,
        name="UppercaseBatch"
    )

    inputs = ["hello", "world", "this", "is", "a", "test"]
    results = processor.process(inputs, show_progress=False)

    expected = ["HELLO", "WORLD", "THIS", "IS", "A", "TEST"]
    assert results == expected, f"Expected {expected}, got {results}"

    print(f"  Input:  {inputs}")
    print(f"  Output: {results}")
    print(f"  ✓ PASSED")


def test_processing_pipeline():
    """Test chained processing pipeline."""
    print("\n" + "="*60)
    print("TEST 3: Processing Pipeline")
    print("="*60)

    from synthesis_enhancements.processing import ProcessingPipeline

    # Create pipeline: double -> add 10 -> square
    pipeline = ProcessingPipeline(show_progress=False)
    pipeline.add_stage("double", lambda x: x * 2)
    pipeline.add_stage("add_ten", lambda x: x + 10)
    pipeline.add_stage("square", lambda x: x ** 2)

    inputs = [1, 2, 3, 4, 5]
    result = pipeline.run(inputs)

    # Formula: ((x * 2) + 10) ** 2
    # 1: ((1*2)+10)^2 = 12^2 = 144
    # 2: ((2*2)+10)^2 = 14^2 = 196
    expected = [144, 196, 256, 324, 400]
    assert result.outputs == expected, f"Expected {expected}, got {result.outputs}"

    print(f"  Pipeline: double -> add_ten -> square")
    print(f"  Formula:  ((x * 2) + 10) ** 2")
    print(f"  Input:    {inputs}")
    print(f"  Output:   {result.outputs}")
    print(f"  Time:     {result.total_time:.4f}s")
    print(f"  ✓ PASSED")


def test_material_parser():
    """Test material formula parsing."""
    print("\n" + "="*60)
    print("TEST 4: Material Parser")
    print("="*60)

    from synthesis_enhancements.parsers import EnhancedMaterialParser

    parser = EnhancedMaterialParser(num_workers=2)
    parser.load(load_embeddings=False)

    test_cases = [
        ("LiCoO2", {"Li": 1.0, "Co": 1.0, "O": 2.0}),
        ("Fe2O3", {"Fe": 2.0, "O": 3.0}),
        ("NaCl", {"Na": 1.0, "Cl": 1.0}),
        ("Li0.5Na0.5MnO2", {"Li": 0.5, "Na": 0.5, "Mn": 1.0, "O": 2.0}),
    ]

    for formula, expected_comp in test_cases:
        result = parser.parse(formula)
        actual_comp = {c.element: c.amount for c in result.composition}

        print(f"\n  Formula: {formula}")
        print(f"  Parsed:  {actual_comp}")

        # Check composition matches
        for elem, amount in expected_comp.items():
            assert elem in actual_comp, f"Missing element {elem}"
            assert abs(actual_comp[elem] - amount) < 0.01, \
                f"Wrong amount for {elem}: expected {amount}, got {actual_comp[elem]}"

        print(f"  ✓ Correct")

    print(f"\n  ✓ ALL {len(test_cases)} CASES PASSED")


def test_material_parser_batch():
    """Test batch material parsing with parallel processing."""
    print("\n" + "="*60)
    print("TEST 5: Batch Material Parsing (Parallel)")
    print("="*60)

    from synthesis_enhancements.parsers import EnhancedMaterialParser

    parser = EnhancedMaterialParser(num_workers=4)
    parser.load(load_embeddings=False)

    materials = [
        "LiCoO2",
        "NaCl",
        "Fe2O3",
        "Al2O3",
        "TiO2",
        "ZnO",
        "CuO",
        "MgO",
    ]

    # Parse in parallel
    results = parser.parse_batch(materials, use_parallel=True)

    print(f"  Parsed {len(results)} materials in parallel:")
    for mat, result in zip(materials, results):
        comp = {c.element: c.amount for c in result.composition}
        print(f"    {mat:12} -> {comp}")

    assert len(results) == len(materials)
    print(f"\n  ✓ PASSED")


def test_phase_detection():
    """Test crystal phase detection."""
    print("\n" + "="*60)
    print("TEST 6: Phase Detection")
    print("="*60)

    from synthesis_enhancements.parsers import EnhancedMaterialParser

    parser = EnhancedMaterialParser()
    parser.load(load_embeddings=False)

    test_cases = [
        ("α-Al2O3", "alpha"),
        ("beta-Li2TiO3", "beta"),
        ("cubic ZrO2", "cubic"),
        ("tetragonal BaTiO3", "tetragonal"),
        ("LiCoO2", None),  # No phase specified
    ]

    for material, expected_phase in test_cases:
        result = parser.parse(material)
        print(f"  {material:20} -> phase: {result.phase}")
        assert result.phase == expected_phase, \
            f"Expected phase '{expected_phase}', got '{result.phase}'"

    print(f"\n  ✓ ALL {len(test_cases)} CASES PASSED")


def test_config_classes():
    """Test configuration classes."""
    print("\n" + "="*60)
    print("TEST 7: Configuration Classes")
    print("="*60)

    from synthesis_enhancements.utils.config import (
        ModelConfig,
        ProcessingConfig,
        ExtractorConfig,
    )

    # Test ModelConfig
    config = ModelConfig(model_name="scibert", max_sequence_length=256)
    assert config.model_name == "scibert"
    assert config.hf_model_name == "allenai/scibert_scivocab_uncased"
    print(f"  ModelConfig: ✓")

    # Test ProcessingConfig
    config = ProcessingConfig(batch_size=64, num_workers=8)
    assert config.batch_size == 64
    assert config.num_workers == 8
    print(f"  ProcessingConfig: ✓")

    # Test ExtractorConfig
    config = ExtractorConfig()
    assert config.confidence_threshold == 0.5
    print(f"  ExtractorConfig: ✓")

    print(f"\n  ✓ ALL PASSED")


def run_all_tests():
    """Run all tests."""
    print("="*60)
    print("SYNTHESIS ENHANCEMENTS - STANDALONE TESTS")
    print("="*60)
    print("Running tests without model downloads...")

    tests = [
        test_parallel_processor,
        test_batch_processor,
        test_processing_pipeline,
        test_material_parser,
        test_material_parser_batch,
        test_phase_detection,
        test_config_classes,
    ]

    passed = 0
    failed = 0

    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"\n  ✗ FAILED: {e}")

    print("\n" + "="*60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("="*60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
