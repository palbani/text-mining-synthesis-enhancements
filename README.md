## Acknowledgments

Built upon the original [text-mined-synthesis](https://github.com/CederGroupHub/text-mined-synthesis_public) project by CederGroup at UC Berkeley.


# Synthesis Enhancements

Enhanced text-mining synthesis package with transformer models, parallel processing, and model versioning.

## Overview

This package provides modernized ML/NLP capabilities for extracting chemical synthesis information from scientific literature, featuring:

- **Transformer-based Models**: MatBERT, SciBERT, and other domain-specific transformers
- **Parallel Processing**: Multi-threaded and multi-process batch processing
- **Model Versioning**: MLflow integration for experiment tracking and model registry
- **ONNX Serialization**: Export models for efficient cross-platform deployment

## Installation

```bash
pip install -e .
```

For development with testing tools:
```bash
pip install -e ".[dev]"
```

For GPU support:
```bash
pip install -e ".[gpu]"
```

## Quick Start

### Operations Extraction

Extract synthesis operations from scientific paragraphs:

```python
from synthesis_enhancements import EnhancedOperationsExtractor

# Initialize and load
extractor = EnhancedOperationsExtractor(
    model_name="matbert",
    num_workers=4
)
extractor.load()

# Extract operations
results = extractor.extract([
    "The mixture was heated at 500°C for 2 hours.",
    "After cooling, the sample was ground and sintered at 1000°C."
])

for result in results:
    for op in result.operations:
        print(f"{op.operation_type}: {op.text}")
        if op.conditions:
            print(f"  Conditions: {op.conditions}")
```

### Material Entity Recognition

Identify materials and their roles in synthesis:

```python
from synthesis_enhancements import EnhancedMaterialEntityRecognizer

recognizer = EnhancedMaterialEntityRecognizer(model_name="matbert")
recognizer.load()

result = recognizer.extract("LiCoO2 was synthesized from Li2CO3 and Co3O4")

print(f"Targets: {[t.text for t in result.targets]}")
print(f"Precursors: {[p.text for p in result.precursors]}")
```

### Material Parsing

Parse chemical formulas and compositions:

```python
from synthesis_enhancements import EnhancedMaterialParser

parser = EnhancedMaterialParser(num_workers=4)
parser.load()

result = parser.parse("Li0.5Na0.5CoO2")
print(f"Formula: {result.formula}")
print(f"Composition: {[(c.element, c.amount) for c in result.composition]}")

# Batch parsing
results = parser.parse_batch([
    "LiCoO2",
    "Li2CO3:Co3O4 (1:1)",
    "La1-xSrxMnO3 (x=0.3)"
])
```

## Parallel Processing

### Batch Processing with Multithreading

```python
from synthesis_enhancements.processing import BatchProcessor, ProcessingConfig

config = ProcessingConfig(
    batch_size=32,
    num_workers=4,
    show_progress=True
)

processor = BatchProcessor(
    process_fn=model.predict,
    config=config
)

# Process with parallel batching
results = processor.process_with_threading(large_dataset)
```

### Processing Pipeline

Chain multiple processing stages:

```python
from synthesis_enhancements.processing import ProcessingPipeline

pipeline = ProcessingPipeline()
pipeline.add_parallel_stage("tokenize", tokenize_fn, num_workers=4)
pipeline.add_batch_stage("classify", classify_fn, batch_size=32)
pipeline.add_stage("postprocess", postprocess_fn)

result = pipeline.run(paragraphs)
```

### Async Processing

For I/O-bound operations:

```python
from synthesis_enhancements.processing import AsyncProcessor

processor = AsyncProcessor(
    process_fn=async_fetch_data,
    max_concurrent=10,
    timeout=30.0
)

results = await processor.process(urls)
```

## Model Versioning

Track experiments with MLflow:

```python
from synthesis_enhancements.versioning import ModelVersionManager

manager = ModelVersionManager(
    experiment_name="synthesis-extraction",
    tracking_uri="mlruns"
)

with manager.start_run("train_classifier"):
    manager.log_params({"learning_rate": 0.001, "epochs": 10})

    # Train model...

    manager.log_metrics({"accuracy": 0.95, "f1": 0.92})
    manager.log_model(model, "classifier", registered_model_name="ops-classifier")
```

## ONNX Export

Export models for efficient deployment:

```python
from synthesis_enhancements.serialization import ONNXSerializer, ONNXInferenceEngine

# Export
serializer = ONNXSerializer()
info = serializer.export_transformer(
    model=classifier,
    output_path="models/classifier.onnx",
    tokenizer=tokenizer
)

# Inference
engine = ONNXInferenceEngine("models/classifier.onnx")
outputs = engine.run(input_ids=tokens, attention_mask=mask)
```

## Supported Models

| Model | ID | Description |
|-------|-----|-------------|
| MatBERT | `matbert` | Materials science BERT |
| SciBERT | `scibert` | Scientific vocabulary BERT |
| PubMedBERT | `pubmedbert` | Biomedical text BERT |
| ChemBERT | `chembert` | Chemistry-focused BERT |
| BERT Base | `bert-base` | General BERT |
| RoBERTa | `roberta` | Robustly optimized BERT |

## Configuration

### Model Configuration

```python
from synthesis_enhancements.utils.config import ModelConfig

config = ModelConfig(
    model_name="matbert",
    max_sequence_length=512,
    device="cuda",  # or "cpu", "auto"
    use_fp16=True,  # Half precision for GPU
)
```

### Processing Configuration

```python
from synthesis_enhancements.utils.config import ProcessingConfig

config = ProcessingConfig(
    batch_size=32,
    num_workers=4,
    use_multiprocessing=True,  # False for threading
    show_progress=True,
    timeout=300.0,
)
```

## Testing

Run the test suite:

```bash
pytest synthesis_enhancements/tests/ -v
```

With coverage:

```bash
pytest synthesis_enhancements/tests/ --cov=synthesis_enhancements --cov-report=html
```

## Project Structure

```
synthesis_enhancements/
├── models/              # Transformer model implementations
│   ├── base.py         # Base classes and factory
│   ├── sequence_classifier.py
│   ├── token_classifier.py
│   └── embeddings.py
├── processing/          # Parallel processing infrastructure
│   ├── parallel.py     # ParallelProcessor, WorkerPool
│   ├── batch.py        # BatchProcessor, StreamingBatchProcessor
│   ├── async_processor.py
│   └── pipeline.py     # ProcessingPipeline
├── versioning/          # Model versioning
│   └── mlflow_manager.py
├── serialization/       # Model export
│   └── onnx_export.py
├── extractors/          # High-level extraction APIs
│   ├── operations.py
│   └── materials.py
├── parsers/             # Material parsing
│   └── material_parser.py
├── utils/               # Utilities
│   ├── config.py
│   ├── constants.py
│   └── logging.py
└── tests/               # Unit tests
```

## License

MIT License

## Citation

If you use this package, please cite:

```bibtex
@software{synthesis_enhancements,
  title = {Synthesis Enhancements: Modern ML/NLP for Materials Synthesis},
  year = {2024},
  url = {https://github.com/CederGroupHub/text-mined-synthesis_public}
}
```

