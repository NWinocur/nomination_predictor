#!/usr/bin/env python3
"""
Pytest unit tests for model loading functionality.
These tests verify that models can be loaded correctly by the Streamlit webapp.
"""

from pathlib import Path
import pickle
import sys
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import pytest
from loguru import logger

# Add the nomination_predictor to the path
sys.path.append(str(Path(__file__).parent.parent))

from nomination_predictor.config import MODELS_DIR


@pytest.fixture(scope="session")
def available_models() -> List[Path]:
    """Fixture to get all available model files."""
    models_path = Path(MODELS_DIR)
    if not models_path.exists():
        pytest.skip(f"Models directory not found: {models_path}")
    
    model_files = list(models_path.glob("*.pkl"))
    if not model_files:
        pytest.skip("No model files found")
    
    return model_files


@pytest.fixture
def load_model_data(request) -> Dict[str, Any]:
    """Fixture to load model data for testing."""
    model_path = request.param
    
    # Try to load the model
    try:
        with open(model_path, "rb") as f:
            model_data = pickle.load(f)
    except AttributeError as e:
        if "ProgressXGBRegressor" in str(e):
            # Try to import the custom class if it exists
            try:
                from nomination_predictor.modeling.train import (
                    ProgressXGBRegressor,  # noqa: F401
                )
                with open(model_path, "rb") as f:
                    model_data = pickle.load(f)
            except ImportError:
                pytest.skip(f"Model contains custom class 'ProgressXGBRegressor' that cannot be imported")
        else:
            raise e
    
    return {
        "model_data": model_data,
        "model_path": model_path,
        "model_filename": model_path.name
    }


class TestModelLoading:
    """Test suite for model loading functionality."""
    
    def test_models_directory_exists(self):
        """Test that the models directory exists."""
        models_path = Path(MODELS_DIR)
        assert models_path.exists(), f"Models directory not found: {models_path}"
    
    def test_model_files_exist(self, available_models):
        """Test that model files exist in the models directory."""
        assert len(available_models) > 0, "No model files found in models directory"
        
        for model_file in available_models:
            assert model_file.exists(), f"Model file does not exist: {model_file}"
            assert model_file.suffix == ".pkl", f"Model file should be .pkl: {model_file}"
    
    @pytest.mark.parametrize("load_model_data", 
                             [pytest.param(model_path, id=model_path.name) 
                              for model_path in Path(MODELS_DIR).glob("*.pkl") if Path(MODELS_DIR).exists()],
                             indirect=True)
    def test_model_pickle_loading(self, load_model_data):
        """Test that model files can be loaded with pickle."""
        model_data = load_model_data["model_data"]
        model_filename = load_model_data["model_filename"]
        
        assert model_data is not None, f"Failed to load model data from {model_filename}"
    
    @pytest.mark.parametrize("load_model_data", 
                             [pytest.param(model_path, id=model_path.name) 
                              for model_path in Path(MODELS_DIR).glob("*.pkl") if Path(MODELS_DIR).exists()],
                             indirect=True)
    def test_model_structure_validation(self, load_model_data):
        """Test that model data has the expected structure."""
        model_data = load_model_data["model_data"]
        model_filename = load_model_data["model_filename"]
        
        if isinstance(model_data, dict):
            # New standardized format
            assert "model" in model_data, f"Model dict missing 'model' key in {model_filename}"
            assert "feature_columns" in model_data, f"Model dict missing 'feature_columns' key in {model_filename}"
            assert "metadata" in model_data, f"Model dict missing 'metadata' key in {model_filename}"
        else:
            # Legacy format - just the model object
            assert hasattr(model_data, "predict"), f"Model object missing predict method in {model_filename}"
    
    @pytest.mark.parametrize("load_model_data", 
                             [pytest.param(model_path, id=model_path.name) 
                              for model_path in Path(MODELS_DIR).glob("*.pkl") if Path(MODELS_DIR).exists()],
                             indirect=True)
    def test_feature_columns_extraction(self, load_model_data):
        """Test that feature columns can be extracted from the model."""
        model_data = load_model_data["model_data"]
        model_filename = load_model_data["model_filename"]
        
        feature_columns = self._extract_feature_columns(model_data)
        
        assert feature_columns is not None, f"Could not extract feature columns from {model_filename}"
        assert len(feature_columns) > 0, f"No feature columns found in {model_filename}"
        assert isinstance(feature_columns, list), f"Feature columns should be a list in {model_filename}"
        
        # Check that feature names are strings
        for feature in feature_columns[:5]:  # Check first 5
            assert isinstance(feature, str), f"Feature name should be string, got {type(feature)} in {model_filename}"
    
    @pytest.mark.parametrize("load_model_data", 
                             [pytest.param(model_path, id=model_path.name) 
                              for model_path in Path(MODELS_DIR).glob("*.pkl") if Path(MODELS_DIR).exists()],
                             indirect=True)
    def test_model_prediction_capability(self, load_model_data):
        """Test that the model can make predictions."""
        model_data = load_model_data["model_data"]
        model_filename = load_model_data["model_filename"]
        
        # Extract model and feature columns
        if isinstance(model_data, dict):
            model = model_data["model"]
        else:
            model = model_data
        
        feature_columns = self._extract_feature_columns(model_data)
        
        # Create dummy input
        dummy_input = pd.DataFrame([{col: 0.0 for col in feature_columns}])
        
        # Test prediction
        try:
            prediction = model.predict(dummy_input)
            assert prediction is not None, f"Prediction returned None for {model_filename}"
            assert len(prediction) == 1, f"Expected 1 prediction, got {len(prediction)} for {model_filename}"
            assert isinstance(prediction[0], (int, float, np.number)), f"Prediction should be numeric for {model_filename}"
        except Exception as e:
            pytest.fail(f"Prediction failed for {model_filename}: {e}")
    
    @pytest.mark.parametrize("load_model_data", 
                             [pytest.param(model_path, id=model_path.name) 
                              for model_path in Path(MODELS_DIR).glob("*.pkl") if Path(MODELS_DIR).exists()],
                             indirect=True)
    def test_model_metadata_presence(self, load_model_data):
        """Test that model metadata is present and well-formed."""
        model_data = load_model_data["model_data"]
        model_filename = load_model_data["model_filename"]
        
        if isinstance(model_data, dict) and "metadata" in model_data:
            metadata = model_data["metadata"]
            assert isinstance(metadata, dict), f"Metadata should be a dict in {model_filename}"
            
            # Check for expected metadata fields
            expected_fields = ["model_name", "timestamp", "datetime_saved", "model_type", "feature_count"]
            for field in expected_fields:
                if field in metadata:
                    assert metadata[field] is not None, f"Metadata field '{field}' is None in {model_filename}"
        else:
            # Legacy format - metadata might not be present
            logger.warning(f"No metadata found in {model_filename} (legacy format)")
    
    def _extract_feature_columns(self, model_data: Any) -> List[str]:
        """Helper method to extract feature columns from model data."""
        if isinstance(model_data, dict):
            feature_columns = model_data.get("feature_columns", [])
            model = model_data.get("model")
        else:
            model = model_data
            feature_columns = getattr(model, "feature_names_in_", [])
        
        # Handle feature_columns being a numpy array or empty
        if hasattr(feature_columns, '__len__') and len(feature_columns) == 0:
            # Try alternative ways to get feature names
            if hasattr(model, 'feature_names_'):
                feature_columns = model.feature_names_
            elif hasattr(model, 'get_booster'):
                try:
                    booster = model.get_booster()
                    feature_columns = booster.feature_names
                except Exception:
                    feature_columns = []
        
        # Convert to list if it's a numpy array
        if hasattr(feature_columns, 'tolist'):
            feature_columns = feature_columns.tolist()
        
        return feature_columns


class TestModelCompatibility:
    """Test suite for model compatibility with webapp requirements."""
    
    @pytest.mark.parametrize("load_model_data", 
                             [pytest.param(model_path, id=model_path.name) 
                              for model_path in Path(MODELS_DIR).glob("*.pkl") if Path(MODELS_DIR).exists()],
                             indirect=True)
    def test_webapp_loading_compatibility(self, load_model_data):
        """Test that models can be loaded using the same logic as the webapp."""
        model_data = load_model_data["model_data"]
        model_filename = load_model_data["model_filename"]
        
        # Simulate webapp loading logic
        if isinstance(model_data, dict):
            model = model_data.get("model")
            feature_columns = model_data.get("feature_columns", [])
            metadata = model_data.get("metadata", {})
        else:
            model = model_data
            feature_columns = getattr(model, "feature_names_in_", [])
            metadata = {}
        
        # Apply webapp's feature column handling logic
        if hasattr(feature_columns, '__len__') and len(feature_columns) == 0:
            if hasattr(model, 'feature_names_'):
                feature_columns = model.feature_names_
            elif hasattr(model, 'get_booster'):
                try:
                    booster = model.get_booster()
                    feature_columns = booster.feature_names
                except Exception:
                    feature_columns = []
        
        # Convert to list if it's a numpy array
        if hasattr(feature_columns, 'tolist'):
            feature_columns = feature_columns.tolist()
        
        # Final webapp validation
        assert feature_columns is not None, f"Webapp would fail: no feature columns for {model_filename}"
        assert len(feature_columns) > 0, f"Webapp would fail: empty feature columns for {model_filename}"
    
    @pytest.mark.parametrize("load_model_data", 
                             [pytest.param(model_path, id=model_path.name) 
                              for model_path in Path(MODELS_DIR).glob("*.pkl") if Path(MODELS_DIR).exists()],
                             indirect=True)
    def test_feature_column_types(self, load_model_data):
        """Test that feature columns are the correct type for webapp widgets."""
        model_data = load_model_data["model_data"]
        model_filename = load_model_data["model_filename"]
        
        feature_columns = self._extract_feature_columns(model_data)
        
        # Check that all feature names are valid strings
        for i, feature in enumerate(feature_columns):
            assert isinstance(feature, str), f"Feature {i} is not a string in {model_filename}: {type(feature)}"
            assert len(feature) > 0, f"Feature {i} is empty string in {model_filename}"
            assert not feature.isspace(), f"Feature {i} is whitespace only in {model_filename}"
    
    def _extract_feature_columns(self, model_data: Any) -> List[str]:
        """Helper method to extract feature columns (same as TestModelLoading)."""
        if isinstance(model_data, dict):
            feature_columns = model_data.get("feature_columns", [])
            model = model_data.get("model")
        else:
            model = model_data
            feature_columns = getattr(model, "feature_names_in_", [])
        
        if hasattr(feature_columns, '__len__') and len(feature_columns) == 0:
            if hasattr(model, 'feature_names_'):
                feature_columns = model.feature_names_
            elif hasattr(model, 'get_booster'):
                try:
                    booster = model.get_booster()
                    feature_columns = booster.feature_names
                except Exception:
                    feature_columns = []
        
        if hasattr(feature_columns, 'tolist'):
            feature_columns = feature_columns.tolist()
        
        return feature_columns


if __name__ == "__main__":
    # Allow running as a script for quick testing
    pytest.main([__file__, "-v"])
