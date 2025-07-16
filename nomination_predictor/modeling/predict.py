"""This file shall contain code to perform inference for an already-existing trained model."""

from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from nomination_predictor.config import MODELS_DIR, PROCESSED_DATA_DIR

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
        print("📊 EXCELLENT: The model's predictions are typically within 30 days of the actual confirmation time.")
        print("🔍 TAKEAWAY: The model has high practical utility for predicting confirmation timelines.")
    elif mae < 60:
        print("📊 GOOD: The model's predictions are typically within 60 days of the actual confirmation time.")
        print("🔍 TAKEAWAY: The model provides valuable insights but has moderate error margins.")
    elif mae < 90:
        print("📊 FAIR: The model's predictions are typically within 90 days of the actual confirmation time.")
        print("🔍 TAKEAWAY: The model offers directional guidance but with substantial uncertainty.")
    else:
        print("📊 NEEDS IMPROVEMENT: The model's predictions have large error margins exceeding 90 days.")
        print("🔍 TAKEAWAY: Consider feature engineering, hyperparameter tuning, or alternative algorithms.")
    
    # Detailed interpretation of R²
    print(f"\n===== R² Score: {r2:.4f} =====")
    if r2 > 0.7:
        print("📊 STRONG: The model explains more than 70% of the variance in confirmation times.")
        print("🔍 TAKEAWAY: The model captures most of the systematic patterns in the data.")
    elif r2 > 0.5:
        print("📊 MODERATE: The model explains between 50-70% of the variance in confirmation times.")
        print("🔍 TAKEAWAY: The model captures significant patterns but misses some factors.")
    elif r2 > 0.3:
        print("📊 FAIR: The model explains between 30-50% of the variance in confirmation times.")
        print("🔍 TAKEAWAY: The model identifies some patterns but misses many important factors.")
    else:
        print("📊 WEAK: The model explains less than 30% of the variance in confirmation times.")
        print("🔍 TAKEAWAY: The model has limited predictive power, consider revisiting features or methodology.")
    
    # Context relative to problem domain
    print("\n===== Interpretation in Context =====")
    print(f"• The average nomination takes {y_train.mean():.0f} days to confirm")
    print(f"• With a standard deviation of {y_train.std():.0f} days")
    print(f"• Our model's error (MAE) is {mae:.0f} days, which is {(mae/y_train.std()*100):.0f}% of the standard deviation")
    print(f"• This means our model {('outperforms' if r2 > 0 else 'underperforms')} a baseline model that always predicts the average")
    
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


