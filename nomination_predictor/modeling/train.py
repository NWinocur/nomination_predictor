"""This file shall contain code to train a model."""

from collections import Counter
from datetime import datetime
import itertools
import os
from pathlib import Path
import pickle

from loguru import logger
from tqdm import tqdm
import typer

from nomination_predictor.config import MODELS_DIR, PROCESSED_DATA_DIR

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


def save_model_with_metadata(model, file_prefix, metadata=None, X_train=None, y_train=None, mae=None, r2=None):
    """Save model with timestamp and metadata"""
    
    # Create a sanitized timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    
    # Generate filename
    filename = MODELS_DIR/f"{file_prefix}_{timestamp}.pkl"
    
    # Default metadata if none provided
    if metadata is None:
        metadata = {}
    
    # Add standard metadata
    metadata.update({
        'timestamp': timestamp,
        'features': X_train.columns.tolist(),
        'n_features': X_train.shape[1],
        'metrics': {
            'mae': float(mae),
            'r2': float(r2)
        }
    })
    
    # Save model and metadata
    with open(filename, 'wb') as f:
        pickle.dump({'model': model, 'metadata': metadata}, f)
    
    logger.info(f"Model saved to {filename} with metadata")
    return filename


if __name__ == "__main__":
    app()
