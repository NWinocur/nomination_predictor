"""
Streamlit webapp for Nomination Confirmation Time Prediction.

This webapp allows users to input nomination scenario parameters and get:
1. Predicted confirmation time
2. Most similar historical case
3. Model complexity information
"""

import os
from pathlib import Path
import pickle
import sys
from typing import Any, Dict, Optional

from loguru import logger
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# Add the parent directory to the path so we can import our modules
sys.path.append(str(Path(__file__).parent))

from nomination_predictor.config import MODELS_DIR, PROCESSED_DATA_DIR
from nomination_predictor.modeling.predict import (
    find_similar_historical_case,
    summarize_model_complexity,
)
from nomination_predictor.modeling.similarity import find_most_similar_case

# Page configuration
st.set_page_config(
    page_title="Nomination Confirmation Time Predictor",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f4e79;
        margin: 1rem 0;
    }
    .similarity-box {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data
def load_available_models():
    """Load list of available trained models."""
    models_path = Path(MODELS_DIR)
    if not models_path.exists():
        return []

    model_files = list(models_path.glob("*.pkl"))
    return [f.name for f in model_files]


@st.cache_data
def load_model_and_data(model_filename: str):
    """Load the selected model and associated data."""
    try:
        model_path = Path(MODELS_DIR) / model_filename
        
        # Handle potential pickle issues with custom classes
        try:
            with open(model_path, "rb") as f:
                logger.info(f"will attempt to load model from {model_path}")
                model_data = pickle.load(f)
                logger.info(f"loaded model from {model_path}")
        except AttributeError as e:
            if "ProgressXGBRegressor" in str(e):
                # Try to import the custom class if it exists
                try:
                    from nomination_predictor.modeling.train import (
                        ProgressXGBRegressor,  # noqa: F401
                    )
                    with open(model_path, "rb") as f:
                        model_data = pickle.load(f)
                        logger.info("loaded model with custom class from %s", model_path)
                except ImportError:
                    st.error("Model contains custom class 'ProgressXGBRegressor' that cannot be imported. Please retrain the model.")
                    return None, None
            else:
                raise e

        # Try to load processed data for similarity search
        processed_data_path = Path(PROCESSED_DATA_DIR) / "feature_engineered.csv"
        if processed_data_path.exists():
            logger.info(f"will attempt to load processed data from {processed_data_path}")
            data = pd.read_csv(processed_data_path)
            logger.info(f"loaded processed data from {processed_data_path}")
        else:
            data = None

        return model_data, data
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None


def get_feature_input_widgets(feature_columns: list) -> Dict[str, Any]:
    """Create input widgets for all features."""
    inputs = {}

    # Organize features into categories for better UX
    demographic_features = [
        col
        for col in feature_columns
        if any(term in col.lower() for term in ["age", "gender", "race", "birth", "death"])
    ]
    education_features = [
        col
        for col in feature_columns
        if any(term in col.lower() for term in ["degree", "education", "school", "university"])
    ]
    position_features = [
        col
        for col in feature_columns
        if any(term in col.lower() for term in ["seat", "court", "position", "level"])
    ]
    political_features = [
        col
        for col in feature_columns
        if any(term in col.lower() for term in ["political", "party", "president", "senate"])
    ]
    timing_features = [
        col
        for col in feature_columns
        if any(term in col.lower() for term in ["date", "days", "year", "time"])
    ]
    other_features = [
        col
        for col in feature_columns
        if col
        not in demographic_features
        + education_features
        + position_features
        + political_features
        + timing_features
    ]

    # Create tabs for different feature categories
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        ["Demographics", "Education", "Position", "Political", "Timing", "Other"]
    )

    with tab1:
        st.subheader("Demographic Information")
        for feature in demographic_features:
            inputs[feature] = create_feature_widget(feature)

    with tab2:
        st.subheader("Education Background")
        for feature in education_features:
            inputs[feature] = create_feature_widget(feature)

    with tab3:
        st.subheader("Position Details")
        for feature in position_features:
            inputs[feature] = create_feature_widget(feature)

    with tab4:
        st.subheader("Political Context")
        for feature in political_features:
            inputs[feature] = create_feature_widget(feature)

    with tab5:
        st.subheader("Timing Information")
        for feature in timing_features:
            inputs[feature] = create_feature_widget(feature)

    with tab6:
        st.subheader("Other Features")
        for feature in other_features:
            inputs[feature] = create_feature_widget(feature)

    return inputs


def create_feature_widget(feature_name: str) -> Any:
    """Create appropriate input widget based on feature name and type."""
    # Common feature patterns and their appropriate widgets
    if "age" in feature_name.lower() and "days" in feature_name.lower():
        return st.number_input(
            f"{feature_name.replace('_', ' ').title()}",
            min_value=0,
            max_value=30000,
            value=15000,
            step=365,
            help="Age in days (approximately 15000 = 41 years old)",
        )
    elif "degree_level" in feature_name.lower():
        return st.selectbox(
            f"{feature_name.replace('_', ' ').title()}",
            options=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            index=1,  # Default to JD (2.0)
            help="1=High School, 2=JD, 3=Masters, 4=PhD, etc.",
        )
    elif "seat_level" in feature_name.lower():
        return st.selectbox(
            f"{feature_name.replace('_', ' ').title()}",
            options=["district", "circuit", "other"],
            index=0,
            help="Level of court position",
        )
    elif "political_era" in feature_name.lower():
        return st.selectbox(
            f"{feature_name.replace('_', ' ').title()}",
            options=["pre‑2013", "2013‑2017", "2017‑2021", "2021‑present"],
            index=0,
            help="Political era when nomination was received",
        )
    elif "boolean" in str(type(feature_name)).lower() or feature_name.lower().startswith("is_"):
        return st.checkbox(f"{feature_name.replace('_', ' ').title()}", value=False)
    elif "year" in feature_name.lower():
        return st.number_input(
            f"{feature_name.replace('_', ' ').title()}",
            min_value=1900,
            max_value=2025,
            value=1995,
            step=1,
        )
    elif "days" in feature_name.lower():
        return st.number_input(
            f"{feature_name.replace('_', ' ').title()}",
            min_value=0,
            max_value=2000,
            value=100,
            step=1,
            help="Number of days",
        )
    else:
        # Default to number input for most features
        return st.number_input(
            f"{feature_name.replace('_', ' ').title()}",
            value=0.0,
            help=f"Enter value for {feature_name}",
        )


def display_prediction_results(prediction: float, model_name: str):
    """Display prediction results in a nice format."""
    st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
    st.markdown("### 🎯 Prediction Results")

    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            label="Predicted Confirmation Time",
            value=f"{prediction:.0f} days",
            help="Estimated number of days from nomination to confirmation",
        )

    with col2:
        st.metric(
            label="Model Used",
            value=model_name.replace(".pkl", "").replace("_", " ").title(),
            help="The trained model used for this prediction",
        )

    # Convert to more readable time units
    weeks = prediction / 7
    months = prediction / 30.44  # Average days per month

    st.write(f"**Alternative time units:**")
    st.write(f"• {weeks:.1f} weeks")
    st.write(f"• {months:.1f} months")

    # Interpretation
    if prediction < 30:
        st.success("🚀 **Very Fast**: This nomination is predicted to be confirmed very quickly!")
    elif prediction < 90:
        st.info("⚡ **Fast**: This nomination is predicted to be confirmed relatively quickly.")
    elif prediction < 180:
        st.warning(
            "⏳ **Moderate**: This nomination may take a moderate amount of time to confirm."
        )
    else:
        st.error("🐌 **Slow**: This nomination is predicted to take a long time to confirm.")

    st.markdown("</div>", unsafe_allow_html=True)


def display_similar_case(similar_case_info: Dict[str, Any]):
    """Display information about the most similar historical case."""
    st.markdown('<div class="similarity-box">', unsafe_allow_html=True)
    st.markdown("### 🔍 Most Similar Historical Case")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="Similarity Score",
            value=f"{similar_case_info['similarity_score']:.3f}",
            help="How similar this case is (0-1, higher is more similar)",
        )

    with col2:
        st.metric(
            label="Actual Outcome",
            value=f"{similar_case_info['actual_outcome']:.0f} days",
            help="How long this historical case actually took",
        )

    with col3:
        st.metric(
            label="Data Source",
            value=similar_case_info["dataset_source"].title(),
            help="Whether this case was in training or test data",
        )

    st.markdown("**Why this case is similar:**")
    st.write(similar_case_info["explanation"])

    # Show some key details about the similar case
    if "similar_case_data" in similar_case_info:
        case_data = similar_case_info["similar_case_data"]

        st.markdown("**Key details of similar case:**")

        # Display relevant fields if they exist
        display_fields = [
            "nominee_name",
            "receiveddate",
            "seat_level_cong_recategorized",
            "received_in_senate_political_era",
            "age_at_nomination_approx_days",
        ]

        for field in display_fields:
            if field in case_data and pd.notna(case_data[field]):
                field_name = field.replace("_", " ").title()
                value = case_data[field]
                if "days" in field and isinstance(value, (int, float)):
                    value = f"{value:.0f} days ({value / 365:.1f} years)"
                st.write(f"• **{field_name}**: {value}")

    st.markdown("</div>", unsafe_allow_html=True)


def main():
    """Main Streamlit app function."""
    st.markdown(
        '<h1 class="main-header">⚖️ Nomination Confirmation Time Predictor</h1>',
        unsafe_allow_html=True,
    )

    st.markdown("""
    This tool predicts how long a judicial nomination will take to be confirmed by the Senate,
    based on historical data and machine learning models. Enter the nomination details below
    to get a prediction and see the most similar historical case.
    """)

    # Sidebar for model selection
    st.sidebar.header("Model Selection")
    available_models = load_available_models()

    if not available_models:
        st.error("No trained models found! Please train a model first.")
        st.stop()
    
    # Auto-refresh controls in sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔄 Auto-Refresh Settings")
    
    enable_autorefresh = st.sidebar.checkbox(
        "Enable Auto-Refresh", 
        value=False,
        help="Automatically refresh the page to show updated data"
    )
    
    if enable_autorefresh:
        refresh_interval = st.sidebar.selectbox(
            "Refresh Interval",
            options=[30, 60, 120, 300, 600],
            index=2,  # Default to 120 seconds (2 minutes)
            format_func=lambda x: f"{x} seconds ({x//60} min)" if x >= 60 else f"{x} seconds",
            help="How often to refresh the page"
        )
        
        max_refreshes = st.sidebar.number_input(
            "Max Refreshes",
            min_value=1,
            max_value=1000,
            value=100,
            help="Maximum number of refreshes before stopping (prevents infinite resource usage)"
        )
        
        # Convert seconds to milliseconds for st_autorefresh
        refresh_count = st_autorefresh(
            interval=refresh_interval * 1000,
            limit=max_refreshes,
            debounce=True,  # Pause refresh when user is interacting
            key="nomination_predictor_autorefresh"
        )
        
        # Show refresh status
        if refresh_count > 0:
            st.sidebar.info(f"🔄 Refreshed {refresh_count}/{max_refreshes} times")
            if refresh_count >= max_refreshes:
                st.sidebar.warning("⚠️ Maximum refreshes reached. Disable and re-enable to continue.")
    
    st.sidebar.markdown("---")

    selected_model = st.sidebar.selectbox(
        "Choose a trained model:",
        available_models,
        help="Select which trained model to use for predictions",
    )

    # Load the selected model
    model_data, historical_data = load_model_and_data(selected_model)

    if model_data is None:
        st.error("Failed to load the selected model.")
        st.stop()

    # Extract model and feature information
    if isinstance(model_data, dict):
        model = model_data.get("model")
        feature_columns = model_data.get("feature_columns", [])
        metadata = model_data.get("metadata", {})
    else:
        # Assume it's just the model
        model = model_data
        feature_columns = getattr(model, "feature_names_in_", [])
        metadata = {}

    # Handle feature_columns being a numpy array or empty
    if hasattr(feature_columns, '__len__') and len(feature_columns) == 0:
        # Try alternative ways to get feature names
        if hasattr(model, 'feature_names_'):
            feature_columns = model.feature_names_
        elif hasattr(model, 'get_booster'):
            # For XGBoost models, try to get feature names from booster
            try:
                booster = model.get_booster()
                feature_columns = booster.feature_names
            except Exception:
                feature_columns = []
        
        # If still no feature columns, try to infer from historical data
        if (not hasattr(feature_columns, '__len__') or len(feature_columns) == 0) and historical_data is not None:
            # Use numeric columns from historical data as a fallback
            numeric_cols = historical_data.select_dtypes(include=[np.number]).columns.tolist()
            # Remove target columns
            target_cols = ['days_nom_to_conf', 'target', 'nid']
            feature_columns = [col for col in numeric_cols if col not in target_cols]
            st.warning(f"Could not determine feature columns from model. Using {len(feature_columns)} numeric columns from historical data as fallback.")
    
    # Final check - convert to list if it's a numpy array
    if hasattr(feature_columns, 'tolist'):
        feature_columns = feature_columns.tolist()
    
    # Check if we have any feature columns
    if not feature_columns or len(feature_columns) == 0:
        st.error("Could not determine feature columns from the model or historical data.")
        st.stop()

    # Display model information
    with st.expander("📊 Model Information", expanded=False):
        st.write(f"**Model Type**: {type(model).__name__}")
        st.write(f"**Number of Features**: {len(feature_columns)}")
        if metadata:
            st.write(f"**Description**: {metadata.get('description', 'No description available')}")
            if "parameters" in metadata:
                st.write(f"**Best Parameters**: {metadata['parameters']}")

        # Show model complexity
        if st.button("Show Model Complexity"):
            st.code("Model Complexity Analysis:")
            # This would call summarize_model_complexity but we'll do it in a simpler way for the webapp
            if hasattr(model, "get_booster"):
                booster = model.get_booster()
                dump = booster.get_dump(with_stats=False)
                n_trees = len(dump)
                n_leaves = sum(l.startswith("leaf") for tree in dump for l in tree.splitlines())
                st.write(f"Trees: {n_trees:,}")
                st.write(f"Leaf nodes: {n_leaves:,}")
                st.write(f"'Parameters' (leaf weights): {n_leaves / 1_000_000_000:.6f} B")

    # Main input section
    st.header("📝 Enter Nomination Details")

    # Get feature inputs
    feature_inputs = get_feature_input_widgets(feature_columns)

    # Prediction button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button(
            "🔮 Predict Confirmation Time", type="primary", use_container_width=True
        )

    if predict_button:
        try:
            # Create input DataFrame
            input_df = pd.DataFrame([feature_inputs])

            # Make prediction
            prediction = model.predict(input_df)[0]

            # Display prediction results
            display_prediction_results(prediction, selected_model)

            # Find similar historical case if data is available
            if historical_data is not None:
                try:
                    # Split historical data (this is a simplified approach)
                    # In practice, you'd want to load the actual train/test split used
                    train_size = int(0.8 * len(historical_data))
                    train_data = historical_data.iloc[:train_size]
                    test_data = historical_data.iloc[train_size:]

                    # Assume target column (adjust as needed)
                    target_col = "days_nom_to_conf"  # or whatever your target column is

                    if target_col in historical_data.columns:
                        X_train = train_data[feature_columns]
                        X_test = test_data[feature_columns]
                        y_train = train_data[target_col]
                        y_test = test_data[target_col]

                        similar_case_info = find_similar_historical_case(
                            query_scenario=feature_inputs,
                            X_train=X_train,
                            X_test=X_test,
                            y_train=y_train,
                            y_test=y_test,
                            trained_model=model,
                            feature_columns=feature_columns,
                        )

                        display_similar_case(similar_case_info)
                    else:
                        st.warning(f"Target column '{target_col}' not found in historical data.")

                except Exception as e:
                    st.warning(f"Could not find similar historical case: {str(e)}")
            else:
                st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                st.warning("⚠️ Historical data not available for similarity comparison.")
                st.markdown("</div>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.exception(e)

    # Footer
    st.markdown("---")
    st.markdown(
        """
    <div style='text-align: center; color: #666; font-size: 0.8em;'>
        <p>Nomination Confirmation Time Predictor | Built with Streamlit</p>
        <p>Predictions are based on historical data and machine learning models. 
        Results should be interpreted as estimates, not guarantees.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
