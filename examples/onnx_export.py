#!/usr/bin/env python3
"""Example: ONNX model export and inference."""

from synthesis_enhancements.serialization import ONNXSerializer, ONNXInferenceEngine
from synthesis_enhancements.utils.config import SerializationConfig


def example_export_model():
    """Example: Export a PyTorch model to ONNX."""
    print("\n" + "="*60)
    print("ONNX Export Example")
    print("="*60)

    print("""
    from synthesis_enhancements.serialization import ONNXSerializer
    from synthesis_enhancements import TransformerModelFactory

    # Create and load model
    classifier = TransformerModelFactory.create_classifier(
        task="operation",
        model_name="matbert",
        num_labels=7
    )
    classifier.load()

    # Configure serialization
    config = SerializationConfig(
        opset_version=14,
        optimize=True,
        quantize=False,  # Set True for INT8 quantization
        dynamic_axes=True,  # Variable batch size
    )

    # Export to ONNX
    serializer = ONNXSerializer(config)
    info = serializer.export_transformer(
        model=classifier,
        output_path="models/operation_classifier.onnx",
        tokenizer=classifier.tokenizer,
        max_length=512,
    )

    print(f"Model exported to: {info.path}")
    print(f"Input names: {info.input_names}")
    print(f"Output names: {info.output_names}")
    """)


def example_onnx_inference():
    """Example: Run inference with ONNX model."""
    print("\n" + "="*60)
    print("ONNX Inference Example")
    print("="*60)

    print("""
    from synthesis_enhancements.serialization import ONNXInferenceEngine
    from transformers import AutoTokenizer
    import numpy as np

    # Load engine and tokenizer
    engine = ONNXInferenceEngine(
        "models/operation_classifier.onnx",
        device="cpu",  # or "cuda" for GPU
        num_threads=4,
    )

    tokenizer = AutoTokenizer.from_pretrained("models/tokenizer")

    # Prepare input
    text = "The mixture was heated at 500°C for 2 hours."
    encoded = tokenizer(
        text,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="np",
    )

    # Run inference
    outputs = engine.run(
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
    )

    # Process outputs
    logits = outputs["logits"]
    predicted_class = np.argmax(logits, axis=-1)
    print(f"Predicted class: {predicted_class}")
    """)


def example_benchmark():
    """Example: Benchmark ONNX model performance."""
    print("\n" + "="*60)
    print("ONNX Benchmark Example")
    print("="*60)

    print("""
    import numpy as np
    from synthesis_enhancements.serialization import ONNXInferenceEngine

    engine = ONNXInferenceEngine("models/classifier.onnx")

    # Create example input
    example_input = {
        "input_ids": np.zeros((1, 512), dtype=np.int64),
        "attention_mask": np.ones((1, 512), dtype=np.int64),
    }

    # Run benchmark
    stats = engine.benchmark(
        input_example=example_input,
        num_iterations=100,
        warmup_iterations=10,
    )

    print(f"Mean latency: {stats['mean_ms']:.2f} ms")
    print(f"Std deviation: {stats['std_ms']:.2f} ms")
    print(f"Min latency: {stats['min_ms']:.2f} ms")
    print(f"Max latency: {stats['max_ms']:.2f} ms")
    print(f"Throughput: {stats['throughput_per_second']:.1f} samples/sec")
    """)


def example_batch_inference():
    """Example: Batch inference with ONNX."""
    print("\n" + "="*60)
    print("ONNX Batch Inference Example")
    print("="*60)

    print("""
    from synthesis_enhancements.serialization import ONNXInferenceEngine
    from synthesis_enhancements.processing import BatchProcessor
    import numpy as np

    engine = ONNXInferenceEngine("models/classifier.onnx")

    def run_batch(inputs):
        # Stack inputs into batch
        input_ids = np.vstack([inp["input_ids"] for inp in inputs])
        attention_mask = np.vstack([inp["attention_mask"] for inp in inputs])

        outputs = engine.run(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # Split outputs back to list
        logits = outputs["logits"]
        return [logits[i] for i in range(len(inputs))]

    # Use BatchProcessor for efficient processing
    processor = BatchProcessor(
        process_fn=run_batch,
        batch_size=32,
    )

    # Process large dataset
    results = processor.process(large_dataset, show_progress=True)
    """)


def example_quantization():
    """Example: Model quantization for faster inference."""
    print("\n" + "="*60)
    print("ONNX Quantization Example")
    print("="*60)

    print("""
    from synthesis_enhancements.serialization import ONNXSerializer
    from synthesis_enhancements.utils.config import SerializationConfig

    # Enable quantization
    config = SerializationConfig(
        opset_version=14,
        optimize=True,
        quantize=True,  # INT8 quantization
    )

    serializer = ONNXSerializer(config)

    # Export with quantization
    info = serializer.export_transformer(
        model=classifier,
        output_path="models/classifier_quantized.onnx",
        tokenizer=tokenizer,
    )

    # The quantized model will be ~4x smaller and faster on CPU
    print(f"Quantized model saved to: {info.path}")

    # Compare sizes
    import os
    original_size = os.path.getsize("models/classifier.onnx")
    quantized_size = os.path.getsize("models/classifier_quantized.onnx")
    print(f"Size reduction: {original_size/quantized_size:.1f}x")
    """)


def main():
    """Run all examples."""
    print("="*60)
    print("ONNX Export and Inference Examples")
    print("="*60)

    example_export_model()
    example_onnx_inference()
    example_benchmark()
    example_batch_inference()
    example_quantization()

    print("\n" + "="*60)
    print("Examples completed!")
    print("="*60)


if __name__ == "__main__":
    main()
