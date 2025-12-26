#!/usr/bin/env python3
"""Example: Model versioning with MLflow."""

from synthesis_enhancements.versioning import (
    ModelVersionManager,
    ExperimentTracker,
    ModelRegistry,
)


def example_experiment_tracking():
    """Example: Track experiments with MLflow."""
    print("\n" + "="*60)
    print("Experiment Tracking Example")
    print("="*60)

    print("""
    # Initialize manager
    manager = ModelVersionManager(
        experiment_name="synthesis-extraction",
        tracking_uri="mlruns"
    )

    # Track a training run
    with manager.start_run("train_operation_classifier"):
        # Log parameters
        manager.log_params({
            "model_name": "matbert",
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 10,
        })

        # Train model...
        # for epoch in range(epochs):
        #     loss = train_epoch(model, data)
        #     manager.log_metric("loss", loss, step=epoch)

        # Log final metrics
        manager.log_metrics({
            "accuracy": 0.95,
            "f1_score": 0.92,
            "precision": 0.94,
            "recall": 0.91,
        })

        # Log model
        manager.log_model(
            model=trained_model,
            artifact_path="model",
            registered_model_name="operation-classifier",
        )

    print("Run completed and logged to MLflow!")
    """)


def example_decorator_tracking():
    """Example: Use decorator for tracking."""
    print("\n" + "="*60)
    print("Decorator-based Tracking Example")
    print("="*60)

    print("""
    from synthesis_enhancements.versioning import ExperimentTracker

    tracker = ExperimentTracker("my-experiment")

    @tracker.track
    def train_model(learning_rate, epochs, batch_size):
        # Training code here...
        model = create_model()
        for epoch in range(epochs):
            loss = train_epoch(model, batch_size)

        accuracy = evaluate(model)
        return {"accuracy": accuracy, "final_loss": loss}

    # Parameters are automatically logged
    # Return dict values are logged as metrics
    result = train_model(
        learning_rate=0.001,
        epochs=10,
        batch_size=32
    )
    """)


def example_model_registry():
    """Example: Use model registry."""
    print("\n" + "="*60)
    print("Model Registry Example")
    print("="*60)

    print("""
    from synthesis_enhancements.versioning import ModelRegistry

    registry = ModelRegistry("mlruns")

    # Register a model
    version = registry.register_model(
        model_uri="runs:/abc123/model",
        name="synthesis-classifier",
        tags={"task": "operation_classification"}
    )

    # Transition to production
    registry.transition_model_version(
        name="synthesis-classifier",
        version=version,
        stage="Production"
    )

    # Load production model
    model = registry.load_model(
        name="synthesis-classifier",
        stage="Production"
    )

    # List all registered models
    models = registry.search_models()
    print(f"Registered models: {models}")
    """)


def example_compare_runs():
    """Example: Compare experiment runs."""
    print("\n" + "="*60)
    print("Compare Runs Example")
    print("="*60)

    print("""
    manager = ModelVersionManager(experiment_name="my-experiment")

    # Search for runs with accuracy > 0.9
    good_runs = manager.search_runs(
        filter_string="metrics.accuracy > 0.9",
        order_by=["metrics.accuracy DESC"],
        max_results=10
    )

    for run in good_runs:
        print(f"Run {run['run_id']}: accuracy={run['metrics.accuracy']}")

    # Get specific run details
    run = manager.get_run("abc123")
    print(f"Parameters: {run.data.params}")
    print(f"Metrics: {run.data.metrics}")
    """)


def main():
    """Run all examples."""
    print("="*60)
    print("Model Versioning Examples")
    print("="*60)

    example_experiment_tracking()
    example_decorator_tracking()
    example_model_registry()
    example_compare_runs()

    print("\n" + "="*60)
    print("Examples completed!")
    print("="*60)


if __name__ == "__main__":
    main()
