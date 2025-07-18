"""This file shall contain code to perform inference for an already-existing trained model."""

from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from tqdm import tqdm
import typer
from xgboost import Booster

from nomination_predictor.config import MODELS_DIR, PROCESSED_DATA_DIR
from nomination_predictor.modeling.similarity import find_most_similar_case

app = typer.Typer()


def predict_model(pipeline, X_test):
    """Predict using the trained model with a simple progress indicator"""
    logger.info(f"Predicting using model on {X_test.shape[0]} samples, {X_test.shape[1]} features")
    with tqdm(total=1, desc="Predicting") as progress_bar:
        predictions = pipeline.predict(X_test)
        progress_bar.update(1)
    logger.info("Prediction completed")
    return predictions


def interpret_results(mae, r2, y_train):
    """
    Interprets the results of a model's predictions based on MAE and R2 metrics.

    Args:
        mae (float): Mean Absolute Error of the model
        r2 (float): R2 metric of the model
        y_train (array-like): Actual values of the training set

    Returns:
    """
    # Detailed interpretation of MAE
    print(f"\n===== Mean Absolute Error (MAE): {mae:.2f} =====")
    if mae < 30:
        print(
            "📊 EXCELLENT: The model's predictions are typically within 30 days of the actual confirmation time."
        )
        print(
            "🔍 TAKEAWAY: The model has high practical utility for predicting confirmation timelines."
        )
    elif mae < 60:
        print(
            "📊 GOOD: The model's predictions are typically within 60 days of the actual confirmation time."
        )
        print("🔍 TAKEAWAY: The model provides valuable insights but has moderate error margins.")
    elif mae < 90:
        print(
            "📊 FAIR: The model's predictions are typically within 90 days of the actual confirmation time."
        )
        print(
            "🔍 TAKEAWAY: The model offers directional guidance but with substantial uncertainty."
        )
    else:
        print(
            "📊 NEEDS IMPROVEMENT: The model's predictions have large error margins exceeding 90 days."
        )
        print(
            "🔍 TAKEAWAY: Consider feature engineering, hyperparameter tuning, or alternative algorithms."
        )

    # Detailed interpretation of R²
    print(f"\n===== R² Score: {r2:.4f} =====")
    if r2 > 0.7:
        print("📊 STRONG: The model explains more than 70% of the variance in confirmation times.")
        print("🔍 TAKEAWAY: The model captures most of the systematic patterns in the data.")
    elif r2 > 0.5:
        print(
            "📊 MODERATE: The model explains between 50-70% of the variance in confirmation times."
        )
        print("🔍 TAKEAWAY: The model captures significant patterns but misses some factors.")
    elif r2 > 0.3:
        print("📊 FAIR: The model explains between 30-50% of the variance in confirmation times.")
        print("🔍 TAKEAWAY: The model identifies some patterns but misses many important factors.")
    else:
        print("📊 WEAK: The model explains less than 30% of the variance in confirmation times.")
        print(
            "🔍 TAKEAWAY: The model has limited predictive power, consider revisiting features or methodology."
        )

    # Context relative to problem domain
    print("\n===== Interpretation in Context =====")
    print(f"• The average nomination takes {y_train.mean():.0f} days to confirm")
    print(f"• With a standard deviation of {y_train.std():.0f} days")
    print(
        f"• Our model's error (MAE) is {mae:.0f} days, which is {(mae / y_train.std() * 100):.0f}% of the standard deviation"
    )
    print(
        f"• This means our model {('outperforms' if r2 > 0 else 'underperforms')} a baseline model that always predicts the average"
    )

    # Actionable next steps
    print("\n===== Recommended Next Steps =====")
    if r2 < 0.3 or mae > 90:
        print("1. Consider feature engineering to identify more predictive variables")
        print("2. Try different algorithms (Random Forest, Neural Networks)")
        print("3. Collect additional data or domain-specific features")
    elif r2 < 0.6:
        print("1. Tune hyperparameters to optimize model performance")
        print("2. Explore feature importance to understand key drivers")
        print("3. Consider ensemble methods to improve predictions")
    else:
        print("1. Focus on model interpretability to understand key drivers")
        print("2. Validate on additional test data to ensure generalizability")
        print("3. Consider deploying the model for practical use")


def summarize_model_complexity(model, model_path: str | Path | None = None) -> None:
    """
    Print a human-readable summary of model size/complexity.

    Supports scikit-learn Pipelines that end with an XGBRegressor (sklearn API),
    pure xgboost.Booster objects, or other tree ensembles that expose
    `estimators_` or `n_estimators`.

    Parameters
    ----------
    model : fitted estimator
    model_path : optional, file on disk so we can show byte size
    """
    # 1. Handle pipelines transparently
    if isinstance(model, Pipeline):
        # assume final step is the estimator
        est = model[-1]
    else:
        est = model

    n_leaves = n_trees = None

    # 2. XGBoost sklearn wrapper ------------------
    if hasattr(est, "get_booster"):
        booster: Booster = est.get_booster()
        dump = booster.get_dump(with_stats=False)
        n_trees = len(dump)
        # each line that starts with a leaf tells us a leaf node
        n_leaves = sum(l.startswith("leaf") for tree in dump for l in tree.splitlines())

    # 3. Native xgboost.Booster --------------------
    elif isinstance(est, Booster):
        dump = est.get_dump(with_stats=False)
        n_trees = len(dump)
        n_leaves = sum(l.startswith("leaf") for tree in dump for l in tree.splitlines())

    # 4. Generic sklearn GradientBoost* -----------
    elif isinstance(est, GradientBoostingRegressor):
        n_trees = est.n_estimators_
        n_leaves = sum(tree.tree_.n_leaves for tree in est.estimators_.ravel())

    # 5. Fallback: count attributes ending with '_' as params
    else:
        n_params = sum(
            v.size
            for k, v in est.__dict__.items()
            if k.endswith("_") and isinstance(v, np.ndarray)
        )
        print(f"Model has {n_params:,} learned array parameters (~{n_params / 1e6:.2f} M).")
        return

    # ---------------------------------------------
    # Convert to “billions of parameters” language
    if n_leaves is not None:
        params_in_billions = n_leaves / 1_000_000_000
        print(
            f"Model complexity:\n"
            f"  • Trees      : {n_trees:,}\n"
            f"  • Leaf nodes : {n_leaves:,}\n"
            f"  • ‘Parameters’ (leaf weights) ≈ "
            f"{params_in_billions:.6f} B"
        )
    if model_path is not None and Path(model_path).exists():
        size_mb = Path(model_path).stat().st_size / (1024**2)
        print(f"  • Serialized size on disk : {size_mb:.1f} MB")


def find_similar_historical_case(
    query_scenario: dict,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    trained_model,
    feature_columns: list = None,
) -> dict:
    """
    Find the most similar historical case for a "your scenario is most similar to..." UI feature.

    Parameters
    ----------
    query_scenario : dict
        Dictionary of feature values for the scenario to analyze
    X_train, X_test : pd.DataFrame
        Training and test feature data
    y_train, y_test : pd.Series
        Training and test target values
    trained_model : fitted estimator
        The trained model to extract feature importances from
    feature_columns : list, optional
        List of feature column names to use

    Returns
    -------
    dict
        Dictionary containing similar case info and explanation
    """
    # Combine target values back with features for context
    train_with_target = X_train.copy()
    train_with_target["target"] = y_train

    test_with_target = X_test.copy()
    test_with_target["target"] = y_test

    # Find most similar case
    similar_case = find_most_similar_case(
        query_row=query_scenario,
        train_data=train_with_target,
        test_data=test_with_target,
        model=trained_model,
        feature_columns=feature_columns,
        metric="cosine",
        return_top_k=1,
        include_similarity_score=True,
    )

    # Format for UI display
    result = {
        "similar_case_data": similar_case.to_dict(),
        "similarity_score": similar_case.get("similarity_score", 0),
        "dataset_source": similar_case.get("_dataset_source", "unknown"),
        "explanation": similar_case.get("_explanation", ""),
        "actual_outcome": similar_case.get("target", "unknown"),
        "predicted_outcome": trained_model.predict(pd.DataFrame([query_scenario]))[0]
        if hasattr(trained_model, "predict")
        else None,
    }

    return result


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    features_path: Path = PROCESSED_DATA_DIR / "test_features.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    predictions_path: Path = PROCESSED_DATA_DIR / "test_predictions.csv",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Performing inference for model...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Inference complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
