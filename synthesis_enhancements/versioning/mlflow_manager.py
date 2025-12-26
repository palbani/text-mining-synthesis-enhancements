"""Model versioning and experiment tracking with MLflow."""

from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Optional, Union
import logging
import json
import os
import time

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature

from synthesis_enhancements.utils.config import VersioningConfig

logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    """Metadata for a model version.

    Attributes:
        name: Model name
        version: Model version
        run_id: MLflow run ID
        stage: Model stage (None, Staging, Production, Archived)
        metrics: Model metrics
        params: Model parameters
        tags: Model tags
        created_at: Creation timestamp
    """
    name: str
    version: str
    run_id: str
    stage: Optional[str] = None
    metrics: dict[str, float] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    tags: dict[str, str] = field(default_factory=dict)
    created_at: Optional[float] = None


class ModelVersionManager:
    """Manager for model versioning and tracking.

    Provides a high-level interface for versioning models,
    tracking experiments, and managing model lifecycle.

    Example:
        >>> manager = ModelVersionManager(
        ...     experiment_name="synthesis-extraction",
        ...     tracking_uri="mlruns"
        ... )
        >>>
        >>> with manager.start_run("train_classifier"):
        ...     manager.log_params({"learning_rate": 0.001})
        ...     model.train(data)
        ...     manager.log_metrics({"accuracy": 0.95})
        ...     manager.log_model(model, "classifier")

    Attributes:
        experiment_name: Name of the MLflow experiment
        tracking_uri: URI for MLflow tracking server
        registry_uri: URI for model registry
    """

    def __init__(
        self,
        experiment_name: str = "synthesis-extraction",
        tracking_uri: str = "mlruns",
        registry_uri: Optional[str] = None,
        config: Optional[VersioningConfig] = None,
    ) -> None:
        """Initialize the model version manager.

        Args:
            experiment_name: Name of the experiment
            tracking_uri: MLflow tracking URI
            registry_uri: Model registry URI (defaults to tracking_uri)
            config: Versioning configuration
        """
        self.config = config or VersioningConfig(
            experiment_name=experiment_name,
            tracking_uri=tracking_uri,
            registry_uri=registry_uri,
        )

        self.experiment_name = self.config.experiment_name
        self.tracking_uri = self.config.tracking_uri
        self.registry_uri = self.config.registry_uri or self.tracking_uri

        self._client: Optional[MlflowClient] = None
        self._current_run: Optional[mlflow.ActiveRun] = None

        self._setup_mlflow()

    def _setup_mlflow(self) -> None:
        """Set up MLflow tracking."""
        mlflow.set_tracking_uri(self.tracking_uri)

        # Create or get experiment
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            mlflow.create_experiment(self.experiment_name)

        mlflow.set_experiment(self.experiment_name)
        self._client = MlflowClient(self.tracking_uri)

        logger.info(
            f"MLflow initialized: experiment='{self.experiment_name}', "
            f"tracking_uri='{self.tracking_uri}'"
        )

    @contextmanager
    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[dict[str, str]] = None,
        nested: bool = False,
    ) -> Iterator[mlflow.ActiveRun]:
        """Start a new MLflow run.

        Args:
            run_name: Name for the run
            tags: Tags to attach to the run
            nested: Whether this is a nested run

        Yields:
            Active MLflow run
        """
        with mlflow.start_run(run_name=run_name, nested=nested) as run:
            self._current_run = run

            if tags:
                mlflow.set_tags(tags)

            try:
                yield run
            finally:
                self._current_run = None

    def log_params(self, params: dict[str, Any]) -> None:
        """Log parameters for the current run.

        Args:
            params: Dictionary of parameter names and values
        """
        # Convert complex types to strings
        logged_params = {}
        for key, value in params.items():
            if isinstance(value, (dict, list)):
                logged_params[key] = json.dumps(value)
            else:
                logged_params[key] = value

        mlflow.log_params(logged_params)

    def log_metrics(
        self,
        metrics: dict[str, float],
        step: Optional[int] = None,
    ) -> None:
        """Log metrics for the current run.

        Args:
            metrics: Dictionary of metric names and values
            step: Optional step number
        """
        mlflow.log_metrics(metrics, step=step)

    def log_metric(
        self,
        key: str,
        value: float,
        step: Optional[int] = None,
    ) -> None:
        """Log a single metric.

        Args:
            key: Metric name
            value: Metric value
            step: Optional step number
        """
        mlflow.log_metric(key, value, step=step)

    def log_model(
        self,
        model: Any,
        artifact_path: str,
        registered_model_name: Optional[str] = None,
        input_example: Optional[Any] = None,
        signature: Optional[Any] = None,
        **kwargs: Any,
    ) -> str:
        """Log a model to MLflow.

        Args:
            model: Model object to log
            artifact_path: Path in artifacts to save model
            registered_model_name: Optional name to register model
            input_example: Example input for signature inference
            signature: Model signature
            **kwargs: Additional arguments for mlflow.pytorch.log_model

        Returns:
            Model URI
        """
        import torch

        # Infer signature if not provided
        if signature is None and input_example is not None:
            try:
                with torch.no_grad():
                    if hasattr(model, "predict"):
                        output = model.predict(input_example)
                    else:
                        output = model(input_example)
                signature = infer_signature(input_example, output)
            except Exception as e:
                logger.warning(f"Could not infer signature: {e}")

        # Log model
        if hasattr(model, "_model"):
            # For wrapped models (SynthesisTransformerModel)
            model_info = mlflow.pytorch.log_model(
                model._model,
                artifact_path,
                registered_model_name=registered_model_name,
                signature=signature,
                input_example=input_example,
                **kwargs,
            )
        else:
            model_info = mlflow.pytorch.log_model(
                model,
                artifact_path,
                registered_model_name=registered_model_name,
                signature=signature,
                input_example=input_example,
                **kwargs,
            )

        logger.info(f"Model logged: {model_info.model_uri}")
        return model_info.model_uri

    def log_artifact(
        self,
        local_path: Union[str, Path],
        artifact_path: Optional[str] = None,
    ) -> None:
        """Log an artifact file.

        Args:
            local_path: Local path to the artifact
            artifact_path: Path in artifacts directory
        """
        mlflow.log_artifact(str(local_path), artifact_path)

    def log_dict(
        self,
        dictionary: dict[str, Any],
        artifact_file: str,
    ) -> None:
        """Log a dictionary as a JSON artifact.

        Args:
            dictionary: Dictionary to log
            artifact_file: Artifact filename
        """
        mlflow.log_dict(dictionary, artifact_file)

    def set_tags(self, tags: dict[str, str]) -> None:
        """Set tags for the current run.

        Args:
            tags: Dictionary of tag names and values
        """
        mlflow.set_tags(tags)

    def get_run(self, run_id: str) -> mlflow.entities.Run:
        """Get a run by ID.

        Args:
            run_id: Run ID

        Returns:
            MLflow Run object
        """
        return self._client.get_run(run_id)

    def search_runs(
        self,
        filter_string: str = "",
        max_results: int = 100,
        order_by: Optional[list[str]] = None,
    ) -> list[mlflow.entities.Run]:
        """Search for runs.

        Args:
            filter_string: Filter query string
            max_results: Maximum number of results
            order_by: Fields to order by

        Returns:
            List of matching runs
        """
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            return []

        return mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=filter_string,
            max_results=max_results,
            order_by=order_by,
        ).to_dict("records")

    def load_model(
        self,
        model_uri: str,
    ) -> Any:
        """Load a model from MLflow.

        Args:
            model_uri: Model URI (e.g., 'runs:/abc123/model')

        Returns:
            Loaded model
        """
        return mlflow.pytorch.load_model(model_uri)


class ExperimentTracker:
    """Simplified experiment tracking interface.

    Provides a decorator-based approach for tracking experiments.

    Example:
        >>> tracker = ExperimentTracker("my-experiment")
        >>>
        >>> @tracker.track
        ... def train_model(learning_rate, epochs):
        ...     model = Model()
        ...     accuracy = model.train(lr=learning_rate, epochs=epochs)
        ...     return {"accuracy": accuracy}
        >>>
        >>> result = train_model(learning_rate=0.001, epochs=10)
    """

    def __init__(
        self,
        experiment_name: str,
        tracking_uri: str = "mlruns",
    ) -> None:
        """Initialize the experiment tracker.

        Args:
            experiment_name: Name of the experiment
            tracking_uri: MLflow tracking URI
        """
        self.manager = ModelVersionManager(
            experiment_name=experiment_name,
            tracking_uri=tracking_uri,
        )

    def track(
        self,
        func: Optional[Callable] = None,
        run_name: Optional[str] = None,
        log_params: bool = True,
        log_result: bool = True,
    ) -> Callable:
        """Decorator to track function execution.

        Args:
            func: Function to track
            run_name: Name for the run
            log_params: Whether to log function arguments
            log_result: Whether to log return value as metrics

        Returns:
            Decorated function
        """
        def decorator(fn: Callable) -> Callable:
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                with self.manager.start_run(run_name=run_name or fn.__name__):
                    # Log parameters from kwargs
                    if log_params:
                        self.manager.log_params(kwargs)

                    # Execute function
                    start_time = time.time()
                    result = fn(*args, **kwargs)
                    duration = time.time() - start_time

                    # Log duration
                    self.manager.log_metric("duration_seconds", duration)

                    # Log result as metrics if it's a dict
                    if log_result and isinstance(result, dict):
                        metrics = {
                            k: v for k, v in result.items()
                            if isinstance(v, (int, float))
                        }
                        if metrics:
                            self.manager.log_metrics(metrics)

                    return result

            return wrapper

        if func is not None:
            return decorator(func)
        return decorator


class ModelRegistry:
    """Interface for the MLflow model registry.

    Manages model versions, stages, and deployment.

    Example:
        >>> registry = ModelRegistry("mlruns")
        >>> registry.register_model("runs:/abc123/model", "my-model")
        >>> registry.transition_model_version("my-model", "1", "Production")
        >>> model = registry.load_model("my-model", stage="Production")
    """

    def __init__(self, tracking_uri: str = "mlruns") -> None:
        """Initialize the model registry.

        Args:
            tracking_uri: MLflow tracking/registry URI
        """
        self.tracking_uri = tracking_uri
        mlflow.set_tracking_uri(tracking_uri)
        self._client = MlflowClient(tracking_uri)

    def register_model(
        self,
        model_uri: str,
        name: str,
        tags: Optional[dict[str, str]] = None,
    ) -> str:
        """Register a model in the registry.

        Args:
            model_uri: Source model URI
            name: Name for the registered model
            tags: Optional tags

        Returns:
            Model version number
        """
        result = mlflow.register_model(model_uri, name)

        if tags:
            for key, value in tags.items():
                self._client.set_model_version_tag(
                    name, result.version, key, value
                )

        logger.info(f"Registered model '{name}' version {result.version}")
        return result.version

    def transition_model_version(
        self,
        name: str,
        version: str,
        stage: str,
        archive_existing: bool = True,
    ) -> None:
        """Transition a model version to a new stage.

        Args:
            name: Model name
            version: Version number
            stage: Target stage (Staging, Production, Archived)
            archive_existing: Whether to archive current version in stage
        """
        self._client.transition_model_version_stage(
            name=name,
            version=version,
            stage=stage,
            archive_existing_versions=archive_existing,
        )
        logger.info(f"Transitioned {name} v{version} to {stage}")

    def get_latest_versions(
        self,
        name: str,
        stages: Optional[list[str]] = None,
    ) -> list[ModelMetadata]:
        """Get latest model versions.

        Args:
            name: Model name
            stages: Filter by stages

        Returns:
            List of model metadata
        """
        versions = self._client.get_latest_versions(name, stages=stages)
        return [
            ModelMetadata(
                name=v.name,
                version=v.version,
                run_id=v.run_id,
                stage=v.current_stage,
                created_at=v.creation_timestamp,
            )
            for v in versions
        ]

    def load_model(
        self,
        name: str,
        version: Optional[str] = None,
        stage: Optional[str] = None,
    ) -> Any:
        """Load a model from the registry.

        Args:
            name: Model name
            version: Specific version (or use stage)
            stage: Stage to load from (Staging, Production)

        Returns:
            Loaded model
        """
        if version:
            model_uri = f"models:/{name}/{version}"
        elif stage:
            model_uri = f"models:/{name}/{stage}"
        else:
            model_uri = f"models:/{name}/latest"

        return mlflow.pytorch.load_model(model_uri)

    def delete_model_version(self, name: str, version: str) -> None:
        """Delete a model version.

        Args:
            name: Model name
            version: Version to delete
        """
        self._client.delete_model_version(name, version)
        logger.info(f"Deleted {name} v{version}")

    def search_models(
        self,
        filter_string: str = "",
        max_results: int = 100,
    ) -> list[str]:
        """Search for registered models.

        Args:
            filter_string: Filter query
            max_results: Maximum results

        Returns:
            List of model names
        """
        models = self._client.search_registered_models(
            filter_string=filter_string,
            max_results=max_results,
        )
        return [m.name for m in models]
