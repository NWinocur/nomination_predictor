"""This file shall contain code to train a model."""

from collections import Counter
from datetime import datetime
import itertools
import os
from pathlib import Path
import pickle
from typing import Any, Dict, Optional, Union

from loguru import logger
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from tqdm import tqdm
import typer

from nomination_predictor.config import MODELS_DIR, PROCESSED_DATA_DIR


def save_model_with_metadata(
    model: Union[BaseEstimator, Pipeline],
    model_name: str,
    feature_columns: list,
    training_data_info: Optional[Dict[str, Any]] = None,
    hyperparameters: Optional[Dict[str, Any]] = None,
    performance_metrics: Optional[Dict[str, Any]] = None,
    custom_metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Save a trained model with comprehensive metadata in a standardized format.
    
    This function ensures all models (tuned or untuned) are saved consistently
    for easy loading by the webapp and other components.
    
    Args:
        model: The trained model (sklearn Pipeline or estimator)
        model_name: Base name for the model (e.g., "xgboost_regression")
        feature_columns: List of feature column names used for training
        training_data_info: Dict with info about training data (shape, target, etc.)
        hyperparameters: Dict of model hyperparameters used
        performance_metrics: Dict of model performance metrics (MAE, R2, etc.)
        custom_metadata: Any additional metadata to include
        
    Returns:
        Path to the saved model file
        
    Example:
        >>> model_path = save_model_with_metadata(
        ...     model=trained_pipeline,
        ...     model_name="xgboost_regression",
        ...     feature_columns=X_train.columns.tolist(),
        ...     training_data_info={
        ...         "n_samples": len(X_train),
        ...         "n_features": len(X_train.columns),
        ...         "target_column": "days_nom_to_conf"
        ...     },
        ...     performance_metrics={
        ...         "train_mae": 45.2,
        ...         "test_mae": 48.7,
        ...         "train_r2": 0.85,
        ...         "test_r2": 0.82
        ...     }
        ... )
    """
    # Create sanitized timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    
    # Create filename
    filename = f"{model_name}_{timestamp}.pkl"
    model_path = Path(MODELS_DIR) / filename
    
    # Ensure models directory exists
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare comprehensive metadata
    metadata = {
        "model_name": model_name,
        "timestamp": timestamp,
        "datetime_saved": datetime.now().isoformat(),
        "model_type": type(model).__name__,
        "sklearn_version": None,  # Will be filled if available
        "feature_count": len(feature_columns),
        "training_data_info": training_data_info or {},
        "hyperparameters": hyperparameters or {},
        "performance_metrics": performance_metrics or {},
        "custom_metadata": custom_metadata or {},
    }
    
    # Try to get sklearn version
    try:
        import sklearn
        metadata["sklearn_version"] = sklearn.__version__
    except (ImportError, AttributeError):
        pass
    
    # Try to get XGBoost version if it's an XGBoost model
    try:
        import xgboost
        if "xgb" in model_name.lower() or "xgboost" in str(type(model)).lower():
            metadata["xgboost_version"] = xgboost.__version__
    except (ImportError, AttributeError):
        pass
    
    # Create the model data structure
    model_data = {
        "model": model,
        "feature_columns": feature_columns,
        "metadata": metadata,
    }
    
    # Save with context manager for safety
    try:
        with open(model_path, "wb") as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved successfully to {model_path}")
        logger.info(f"Model metadata: {metadata['model_name']} with {metadata['feature_count']} features")
        
        return model_path
        
    except Exception as e:
        logger.error(f"Failed to save model to {model_path}: {e}")
        raise


def load_model_with_metadata(model_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a model saved with save_model_with_metadata().
    
    Args:
        model_path: Path to the saved model file
        
    Returns:
        Dict containing 'model', 'feature_columns', and 'metadata'
        
    Example:
        >>> model_data = load_model_with_metadata("models/xgboost_regression_2025-01-15_143022.pkl")
        >>> model = model_data["model"]
        >>> features = model_data["feature_columns"]
        >>> metadata = model_data["metadata"]
    """
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    try:
        with open(model_path, "rb") as f:
            model_data = pickle.load(f)
        
        # Validate structure
        required_keys = ["model", "feature_columns", "metadata"]
        if not all(key in model_data for key in required_keys):
            raise ValueError(f"Model file missing required keys: {required_keys}")
        
        logger.info(f"Model loaded successfully from {model_path}")
        return model_data
        
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {e}")
        raise


app = typer.Typer()


def validate_feature_lists(numeric_features, boolean_features, categorical_features):
    """
    Validates that there are no duplicate features across the different feature type lists.

    Args:
        numeric_features (list): List of numeric feature names
        boolean_features (list): List of boolean feature names
        categorical_features (list): List of categorical feature names

    Returns:
        bool: True if there are no duplicates, False otherwise
    """

    # Combine all features
    all_features = list(itertools.chain(numeric_features, boolean_features, categorical_features))

    # Count occurrences of each feature
    feature_counts = Counter(all_features)

    # Find duplicates
    duplicates = [feature for feature, count in feature_counts.items() if count > 1]

    if duplicates:
        print("⚠️ DUPLICATE FEATURES DETECTED:")
        for dup in duplicates:
            print(f"  - '{dup}' appears in multiple feature lists:")
            if dup in numeric_features:
                print("    • numeric_features")
            if dup in boolean_features:
                print("    • boolean_features")
            if dup in categorical_features:
                print("    • categorical_features")
        return False
    else:
        print("✅ All features are unique across feature type lists")

        # Additionally, check total coverage
        all_features_set = set(all_features)
        if not all_features_set:
            print("⚠️ WARNING: No features specified in any list")
        else:
            print(f"ℹ️ Total unique features: {len(all_features_set)}")

        return True


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Training some model...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Modeling training complete.")
    # -----------------------------------------


def train_model(pipeline, X_train, y_train):
    """Train the model with a simple progress indicator"""
    logger.info(f"Training model on {X_train.shape[0]} samples, {X_train.shape[1]} features")
    with tqdm(total=1, desc="Training Pipeline") as progress_bar:
        pipeline.fit(X_train, y_train)
        progress_bar.update(1)
    logger.info("Model training completed")
    return pipeline


if __name__ == "__main__":
    app()
