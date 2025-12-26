"""ONNX model serialization and inference."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union
import logging
import json

import numpy as np
import torch
import torch.nn as nn

from synthesis_enhancements.utils.config import SerializationConfig

logger = logging.getLogger(__name__)


@dataclass
class ONNXModelInfo:
    """Information about an exported ONNX model.

    Attributes:
        path: Path to the ONNX file
        input_names: Names of input tensors
        output_names: Names of output tensors
        input_shapes: Shapes of input tensors
        output_shapes: Shapes of output tensors
        opset_version: ONNX opset version
        metadata: Additional metadata
    """
    path: Path
    input_names: list[str]
    output_names: list[str]
    input_shapes: dict[str, tuple]
    output_shapes: dict[str, tuple]
    opset_version: int
    metadata: dict[str, Any]


class ONNXSerializer:
    """Serializer for exporting models to ONNX format.

    Supports exporting PyTorch models to ONNX with optional
    optimization and quantization.

    Example:
        >>> serializer = ONNXSerializer()
        >>> info = serializer.export(
        ...     model=classifier,
        ...     output_path="model.onnx",
        ...     input_example=sample_input
        ... )
        >>> print(f"Exported to: {info.path}")

    Attributes:
        config: Serialization configuration
    """

    def __init__(
        self,
        config: Optional[SerializationConfig] = None,
    ) -> None:
        """Initialize the ONNX serializer.

        Args:
            config: Serialization configuration
        """
        self.config = config or SerializationConfig()

    def export(
        self,
        model: nn.Module,
        output_path: Union[str, Path],
        input_example: Union[torch.Tensor, dict[str, torch.Tensor]],
        input_names: Optional[list[str]] = None,
        output_names: Optional[list[str]] = None,
        dynamic_axes: Optional[dict[str, dict[int, str]]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> ONNXModelInfo:
        """Export a PyTorch model to ONNX format.

        Args:
            model: PyTorch model to export
            output_path: Output file path
            input_example: Example input tensor(s)
            input_names: Names for input tensors
            output_names: Names for output tensors
            dynamic_axes: Dynamic axes specification
            metadata: Additional metadata to save

        Returns:
            Information about the exported model
        """
        import onnx

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Set model to eval mode
        model.eval()

        # Prepare input names
        if input_names is None:
            if isinstance(input_example, dict):
                input_names = list(input_example.keys())
            else:
                input_names = ["input"]

        # Prepare output names
        if output_names is None:
            output_names = ["output"]

        # Prepare dynamic axes
        if dynamic_axes is None and self.config.dynamic_axes:
            dynamic_axes = self._create_dynamic_axes(input_names, output_names)

        # Prepare input tuple
        if isinstance(input_example, dict):
            args = tuple(input_example.values())
        else:
            args = (input_example,)

        # Export to ONNX
        logger.info(f"Exporting model to ONNX: {output_path}")

        torch.onnx.export(
            model,
            args,
            str(output_path),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=self.config.opset_version,
            do_constant_folding=True,
            export_params=True,
        )

        # Load and validate
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)

        # Optimize if requested
        if self.config.optimize:
            output_path = self._optimize_model(output_path, onnx_model)

        # Quantize if requested
        if self.config.quantize:
            output_path = self._quantize_model(output_path)

        # Save metadata
        info = self._create_model_info(
            output_path,
            input_names,
            output_names,
            input_example,
            metadata or {},
        )
        self._save_metadata(output_path, info)

        logger.info(f"Model exported successfully: {output_path}")
        return info

    def _create_dynamic_axes(
        self,
        input_names: list[str],
        output_names: list[str],
    ) -> dict[str, dict[int, str]]:
        """Create dynamic axes for variable batch size.

        Args:
            input_names: Input tensor names
            output_names: Output tensor names

        Returns:
            Dynamic axes specification
        """
        dynamic_axes = {}

        for name in input_names:
            dynamic_axes[name] = {0: "batch_size"}

        for name in output_names:
            dynamic_axes[name] = {0: "batch_size"}

        return dynamic_axes

    def _optimize_model(
        self,
        path: Path,
        onnx_model: Any,
    ) -> Path:
        """Optimize the ONNX model.

        Args:
            path: Model path
            onnx_model: ONNX model

        Returns:
            Path to optimized model
        """
        try:
            from onnxruntime.transformers import optimizer
            from onnxruntime.transformers.fusion_options import FusionOptions

            # Create optimization options
            options = FusionOptions("bert")
            options.enable_skip_layer_norm = True
            options.enable_embed_layer_norm = True

            optimized_model = optimizer.optimize_model(
                str(path),
                model_type="bert",
                num_heads=12,
                hidden_size=768,
                optimization_options=options,
            )

            optimized_path = path.with_suffix(".optimized.onnx")
            optimized_model.save_model_to_file(str(optimized_path))

            logger.info(f"Model optimized: {optimized_path}")
            return optimized_path

        except ImportError:
            logger.warning("onnxruntime-tools not installed, skipping optimization")
            return path
        except Exception as e:
            logger.warning(f"Optimization failed: {e}")
            return path

    def _quantize_model(self, path: Path) -> Path:
        """Quantize the ONNX model to INT8.

        Args:
            path: Model path

        Returns:
            Path to quantized model
        """
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType

            quantized_path = path.with_suffix(".quantized.onnx")

            quantize_dynamic(
                str(path),
                str(quantized_path),
                weight_type=QuantType.QInt8,
            )

            logger.info(f"Model quantized: {quantized_path}")
            return quantized_path

        except ImportError:
            logger.warning("onnxruntime quantization not available")
            return path
        except Exception as e:
            logger.warning(f"Quantization failed: {e}")
            return path

    def _create_model_info(
        self,
        path: Path,
        input_names: list[str],
        output_names: list[str],
        input_example: Any,
        metadata: dict[str, Any],
    ) -> ONNXModelInfo:
        """Create model information object.

        Args:
            path: Model path
            input_names: Input names
            output_names: Output names
            input_example: Example input
            metadata: Additional metadata

        Returns:
            Model information
        """
        # Get input shapes
        input_shapes = {}
        if isinstance(input_example, dict):
            for name, tensor in input_example.items():
                input_shapes[name] = tuple(tensor.shape)
        else:
            input_shapes[input_names[0]] = tuple(input_example.shape)

        return ONNXModelInfo(
            path=path,
            input_names=input_names,
            output_names=output_names,
            input_shapes=input_shapes,
            output_shapes={},  # Determined at runtime
            opset_version=self.config.opset_version,
            metadata=metadata,
        )

    def _save_metadata(self, path: Path, info: ONNXModelInfo) -> None:
        """Save model metadata to JSON file.

        Args:
            path: Model path
            info: Model information
        """
        metadata_path = path.with_suffix(".json")
        metadata = {
            "input_names": info.input_names,
            "output_names": info.output_names,
            "input_shapes": {k: list(v) for k, v in info.input_shapes.items()},
            "opset_version": info.opset_version,
            **info.metadata,
        }

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    def export_transformer(
        self,
        model: Any,
        output_path: Union[str, Path],
        tokenizer: Any,
        max_length: int = 512,
        metadata: Optional[dict[str, Any]] = None,
    ) -> ONNXModelInfo:
        """Export a transformer model with tokenizer info.

        Specialized export for HuggingFace transformer models.

        Args:
            model: Transformer model (or SynthesisTransformerModel)
            output_path: Output path
            tokenizer: Model tokenizer
            max_length: Maximum sequence length
            metadata: Additional metadata

        Returns:
            Model information
        """
        output_path = Path(output_path)

        # Get underlying model if wrapped
        if hasattr(model, "_model"):
            pytorch_model = model._model
        else:
            pytorch_model = model

        # Create example input
        example_text = "This is an example input for ONNX export."
        encoded = tokenizer(
            example_text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Move to same device as model
        device = next(pytorch_model.parameters()).device
        input_example = {k: v.to(device) for k, v in encoded.items()}

        # Prepare metadata
        export_metadata = {
            "model_type": "transformer",
            "max_length": max_length,
            "tokenizer_class": tokenizer.__class__.__name__,
            **(metadata or {}),
        }

        # Export
        info = self.export(
            model=pytorch_model,
            output_path=output_path,
            input_example=input_example,
            input_names=list(input_example.keys()),
            output_names=["logits"],
            metadata=export_metadata,
        )

        # Save tokenizer
        tokenizer_path = output_path.parent / "tokenizer"
        tokenizer.save_pretrained(str(tokenizer_path))
        logger.info(f"Tokenizer saved to: {tokenizer_path}")

        return info


class ONNXInferenceEngine:
    """Inference engine for ONNX models.

    Provides efficient inference using ONNX Runtime with
    support for CPU and GPU execution.

    Example:
        >>> engine = ONNXInferenceEngine("model.onnx")
        >>> outputs = engine.run(input_ids=token_ids, attention_mask=mask)
        >>> predictions = outputs["logits"]

    Attributes:
        model_path: Path to ONNX model
        session: ONNX Runtime inference session
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        device: str = "cpu",
        num_threads: Optional[int] = None,
    ) -> None:
        """Initialize the inference engine.

        Args:
            model_path: Path to ONNX model
            device: Execution device ('cpu' or 'cuda')
            num_threads: Number of threads for CPU execution
        """
        import onnxruntime as ort

        self.model_path = Path(model_path)

        # Configure session options
        session_options = ort.SessionOptions()

        if num_threads is not None:
            session_options.intra_op_num_threads = num_threads
            session_options.inter_op_num_threads = num_threads

        # Select execution provider
        if device == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers():
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        # Create session
        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=session_options,
            providers=providers,
        )

        # Get input/output names
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]

        logger.info(
            f"ONNX session created: inputs={self.input_names}, "
            f"outputs={self.output_names}, providers={providers}"
        )

    def run(
        self,
        **inputs: Union[np.ndarray, torch.Tensor],
    ) -> dict[str, np.ndarray]:
        """Run inference on inputs.

        Args:
            **inputs: Input tensors by name

        Returns:
            Dictionary mapping output names to arrays
        """
        # Convert torch tensors to numpy
        numpy_inputs = {}
        for name, tensor in inputs.items():
            if isinstance(tensor, torch.Tensor):
                numpy_inputs[name] = tensor.cpu().numpy()
            else:
                numpy_inputs[name] = tensor

        # Run inference
        outputs = self.session.run(
            self.output_names,
            numpy_inputs,
        )

        return dict(zip(self.output_names, outputs))

    def run_batch(
        self,
        inputs_list: list[dict[str, np.ndarray]],
    ) -> list[dict[str, np.ndarray]]:
        """Run inference on multiple inputs.

        Args:
            inputs_list: List of input dictionaries

        Returns:
            List of output dictionaries
        """
        return [self.run(**inputs) for inputs in inputs_list]

    def benchmark(
        self,
        input_example: dict[str, np.ndarray],
        num_iterations: int = 100,
        warmup_iterations: int = 10,
    ) -> dict[str, float]:
        """Benchmark inference performance.

        Args:
            input_example: Example input
            num_iterations: Number of benchmark iterations
            warmup_iterations: Number of warmup iterations

        Returns:
            Benchmark statistics
        """
        import time

        # Warmup
        for _ in range(warmup_iterations):
            self.run(**input_example)

        # Benchmark
        times = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            self.run(**input_example)
            times.append(time.perf_counter() - start)

        times = np.array(times)
        return {
            "mean_ms": float(times.mean() * 1000),
            "std_ms": float(times.std() * 1000),
            "min_ms": float(times.min() * 1000),
            "max_ms": float(times.max() * 1000),
            "throughput_per_second": float(1.0 / times.mean()),
        }

    @classmethod
    def from_transformer(
        cls,
        model_dir: Union[str, Path],
        device: str = "cpu",
    ) -> tuple["ONNXInferenceEngine", Any]:
        """Load an engine from an exported transformer model.

        Args:
            model_dir: Directory containing model and tokenizer
            device: Execution device

        Returns:
            Tuple of (engine, tokenizer)
        """
        from transformers import AutoTokenizer

        model_dir = Path(model_dir)

        # Find ONNX file
        onnx_files = list(model_dir.glob("*.onnx"))
        if not onnx_files:
            raise FileNotFoundError(f"No ONNX files found in {model_dir}")

        model_path = onnx_files[0]

        # Load tokenizer
        tokenizer_path = model_dir / "tokenizer"
        if tokenizer_path.exists():
            tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
        else:
            raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")

        engine = cls(model_path, device=device)
        return engine, tokenizer
