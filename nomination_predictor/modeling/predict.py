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


def train_and_evaluate_model(
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    model_name: str = "model",
    show_progress: bool = True
) -> dict:
    """
    Train a model and evaluate it with comprehensive metrics and reporting.
    
    Args:
        model: Untrained model or pipeline
        X_train, X_test: Training and test features
        y_train, y_test: Training and test targets
        model_name: Name for display purposes
        show_progress: Whether to show progress bars
        
    Returns:
        dict: Contains trained model, predictions, and metrics
    """
    import time

    from sklearn.metrics import mean_absolute_error, r2_score
    
    logger.info(f"Training {model_name} on {X_train.shape[0]} samples, {X_train.shape[1]} features")
    
    # Training
    start_time = time.time()
    if show_progress:
        with tqdm(total=1, desc=f"Training {model_name}") as pbar:
            model.fit(X_train, y_train)
            pbar.update(1)
    else:
        model.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds")
    
    # Predictions
    if show_progress:
        with tqdm(total=2, desc="Making predictions") as pbar:
            y_train_pred = model.predict(X_train)
            pbar.update(1)
            y_test_pred = model.predict(X_test)
            pbar.update(1)
    else:
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    # Return comprehensive results
    return {
        "model": model,
        "training_time": training_time,
        "predictions": {
            "y_train_pred": y_train_pred,
            "y_test_pred": y_test_pred
        },
        "metrics": {
            "train_mae": train_mae,
            "test_mae": test_mae,
            "train_r2": train_r2,
            "test_r2": test_r2
        },
        "data_info": {
            "n_train_samples": len(X_train),
            "n_test_samples": len(X_test),
            "n_features": len(X_train.columns),
            "feature_columns": X_train.columns.tolist()
        }
    }


def evaluate_and_report_model(
    results: dict,
    y_train: pd.Series,
    y_test: pd.Series,
    model_name: str = "Model",
    save_model: bool = False,
    hyperparameters: dict = None
) -> dict:
    """
    Comprehensive model evaluation and reporting with optional model saving.
    
    Args:
        results: Output from train_and_evaluate_model()
        y_train, y_test: Training and test targets
        model_name: Name for display and saving
        save_model: Whether to save the model with metadata
        hyperparameters: Model hyperparameters for metadata
        
    Returns:
        dict: Enhanced results with interpretation and model path (if saved)
    """
    from nomination_predictor.modeling.train import save_model_with_metadata
    
    metrics = results["metrics"]
    model = results["model"]
    
    # Print comprehensive evaluation
    print(f"\n{'='*60}")
    print(f"📊 {model_name.upper()} EVALUATION RESULTS")
    print(f"{'='*60}")
    
    # Performance metrics
    print(f"\n🎯 PERFORMANCE METRICS:")
    print(f"  • Training MAE: {metrics['train_mae']:.2f} days")
    print(f"  • Test MAE:     {metrics['test_mae']:.2f} days")
    print(f"  • Training R²:  {metrics['train_r2']:.4f}")
    print(f"  • Test R²:      {metrics['test_r2']:.4f}")
    print(f"  • Training time: {results['training_time']:.2f} seconds")
    
    # Interpret results using existing function
    interpret_results(metrics['test_mae'], metrics['test_r2'], y_train)
    
    # Model complexity
    print(f"\n🔧 MODEL COMPLEXITY:")
    summarize_model_complexity(model)
    
    # Save model if requested
    model_path = None
    if save_model:
        model_path = save_model_with_metadata(
            model=model,
            model_name=model_name.lower().replace(" ", "_"),
            feature_columns=results["data_info"]["feature_columns"],
            training_data_info={
                "n_train_samples": results["data_info"]["n_train_samples"],
                "n_test_samples": results["data_info"]["n_test_samples"],
                "n_features": results["data_info"]["n_features"],
                "target_column": "days_nom_to_conf",
                "training_time_seconds": results["training_time"]
            },
            hyperparameters=hyperparameters or {},
            performance_metrics=metrics,
            custom_metadata={
                "model_type": "trained_model",
                "evaluation_metrics": "MAE (Mean Absolute Error) and R²",
                "notes": f"Model trained and evaluated using comprehensive workflow"
            }
        )
        print(f"\n💾 Model saved to: {model_path}")
    
    # Return enhanced results
    enhanced_results = results.copy()
    enhanced_results["model_path"] = model_path
    enhanced_results["interpretation"] = {
        "mae_category": _categorize_mae(metrics['test_mae']),
        "r2_category": _categorize_r2(metrics['test_r2']),
        "overall_quality": _assess_overall_quality(metrics['test_mae'], metrics['test_r2'])
    }
    
    return enhanced_results


def _categorize_mae(mae: float) -> str:
    """Categorize MAE performance."""
    if mae < 30:
        return "excellent"
    elif mae < 60:
        return "good"
    elif mae < 90:
        return "fair"
    else:
        return "needs_improvement"


def _categorize_r2(r2: float) -> str:
    """Categorize R² performance."""
    if r2 > 0.7:
        return "strong"
    elif r2 > 0.5:
        return "moderate"
    elif r2 > 0.3:
        return "fair"
    else:
        return "weak"


def _assess_overall_quality(mae: float, r2: float) -> str:
    """Assess overall model quality."""
    mae_cat = _categorize_mae(mae)
    r2_cat = _categorize_r2(r2)
    
    if mae_cat == "excellent" and r2_cat in ["strong", "moderate"]:
        return "production_ready"
    elif mae_cat in ["excellent", "good"] and r2_cat in ["strong", "moderate", "fair"]:
        return "good_performance"
    elif mae_cat in ["good", "fair"] and r2_cat in ["moderate", "fair"]:
        return "acceptable"
    else:
        return "needs_improvement"


def interpret_results(mae, r2, y_train):
    """
    Interprets the results of a model's predictions based on MAE and R2 metrics.

    Args:
        mae (float): Mean Absolute Error of the model
        r2 (float): R2 metric of the model
        y_train (array-like): Actual values of the training set
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


def compare_models(results_list: list, model_names: list) -> None:
    """
    Compare multiple model results and display a comparison table.
    
    Args:
        results_list: List of results from train_and_evaluate_model()
        model_names: List of model names for display
    """
    print(f"\n{'='*80}")
    print(f"📊 MODEL COMPARISON")
    print(f"{'='*80}")
    
    # Create comparison table
    print(f"{'Model':<20} {'Train MAE':<12} {'Test MAE':<12} {'Train R²':<10} {'Test R²':<10} {'Time (s)':<10}")
    print("-" * 80)
    
    best_test_mae = float('inf')
    best_model_name = ""
    
    for results, name in zip(results_list, model_names):
        metrics = results["metrics"]
        time_taken = results["training_time"]
        
        print(f"{name:<20} {metrics['train_mae']:<12.2f} {metrics['test_mae']:<12.2f} "
              f"{metrics['train_r2']:<10.4f} {metrics['test_r2']:<10.4f} {time_taken:<10.2f}")
        
        if metrics['test_mae'] < best_test_mae:
            best_test_mae = metrics['test_mae']
            best_model_name = name
    
    print("-" * 80)
    print(f"🏆 Best model: {best_model_name} (Test MAE: {best_test_mae:.2f} days)")
    
    # Calculate improvements
    if len(results_list) >= 2:
        baseline_mae = results_list[0]["metrics"]["test_mae"]
        best_mae = min(r["metrics"]["test_mae"] for r in results_list)
        improvement = ((baseline_mae - best_mae) / baseline_mae) * 100
        print(f"📈 Improvement over baseline: {improvement:.1f}% ({baseline_mae:.2f} → {best_mae:.2f} days)")


def find_similar_historical_case(
    query_scenario: dict,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    trained_model,
    feature_columns: list = None,
    train_data_full: pd.DataFrame = None,
    test_data_full: pd.DataFrame = None,
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
    train_data_full, test_data_full : pd.DataFrame, optional
        Full training and test data with all columns (including fjc_biography_url)

    Returns
    -------
    dict
        Dictionary containing similar case info and explanation
    """
    # Use full data if provided, otherwise fall back to feature data
    if train_data_full is not None and test_data_full is not None:
        train_with_target = train_data_full.copy()
        test_with_target = test_data_full.copy()
        # Ensure target column exists
        if "target" not in train_with_target.columns:
            train_with_target["target"] = y_train
        if "target" not in test_with_target.columns:
            test_with_target["target"] = y_test
    else:
        # Fallback to original behavior
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
