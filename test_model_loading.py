#!/usr/bin/env python3
"""
Test script to verify model loading works correctly for the Streamlit webapp.
This helps debug model loading issues before running the full webapp.
"""

from pathlib import Path
import pickle
import sys

from loguru import logger
import numpy as np
import pandas as pd

# Add the nomination_predictor to the path
sys.path.append(str(Path(__file__).parent))

from nomination_predictor.config import MODELS_DIR, PROCESSED_DATA_DIR


def test_model_loading(model_filename: str):
    """Test loading a specific model file."""
    print(f"\n🔍 Testing model: {model_filename}")
    print("=" * 50)
    
    try:
        model_path = Path(MODELS_DIR) / model_filename
        
        # Try to load the model
        try:
            with open(model_path, "rb") as f:
                print(f"📂 Loading model from: {model_path}")
                model_data = pickle.load(f)
                print("✅ Model loaded successfully")
        except AttributeError as e:
            if "ProgressXGBRegressor" in str(e):
                print("⚠️  Model contains custom ProgressXGBRegressor class")
                try:
                    from nomination_predictor.modeling.train import (
                        ProgressXGBRegressor,  # noqa: F401
                    )
                    with open(model_path, "rb") as f:
                        model_data = pickle.load(f)
                        print("✅ Model loaded with custom class")
                except ImportError:
                    print("❌ Cannot import ProgressXGBRegressor - model incompatible")
                    return False
            else:
                print(f"❌ AttributeError: {e}")
                return False
        
        # Analyze the model structure
        print(f"📊 Model type: {type(model_data)}")
        
        if isinstance(model_data, dict):
            print("📋 Model is a dictionary with keys:")
            for key in model_data.keys():
                print(f"   - {key}: {type(model_data[key])}")
            
            model = model_data.get("model")
            feature_columns = model_data.get("feature_columns", [])
            metadata = model_data.get("metadata", {})
        else:
            print("📋 Model is a direct object")
            model = model_data
            feature_columns = getattr(model, "feature_names_in_", [])
            metadata = {}
        
        print(f"🤖 Actual model type: {type(model)}")
        print(f"📝 Metadata keys: {list(metadata.keys()) if metadata else 'None'}")
        
        # Test feature columns extraction
        print("\n🔧 Testing feature column extraction:")
        
        # Handle feature_columns being a numpy array or empty
        if hasattr(feature_columns, '__len__') and len(feature_columns) == 0:
            print("⚠️  No feature columns found in model data")
            
            # Try alternative ways to get feature names
            if hasattr(model, 'feature_names_'):
                feature_columns = model.feature_names_
                print(f"✅ Found feature_names_: {len(feature_columns)} features")
            elif hasattr(model, 'get_booster'):
                try:
                    booster = model.get_booster()
                    feature_columns = booster.feature_names
                    print(f"✅ Found booster feature names: {len(feature_columns)} features")
                except Exception as e:
                    print(f"❌ Could not get booster feature names: {e}")
                    feature_columns = []
        
        # Final check - convert to list if it's a numpy array
        if hasattr(feature_columns, 'tolist'):
            feature_columns = feature_columns.tolist()
            print("🔄 Converted numpy array to list")
        
        # Check if we have any feature columns
        if not feature_columns or len(feature_columns) == 0:
            print("❌ No feature columns available")
            return False
        else:
            print(f"✅ Feature columns available: {len(feature_columns)}")
            print(f"   First 5 features: {feature_columns[:5]}")
        
        # Test prediction capability
        print("\n🎯 Testing prediction capability:")
        try:
            # Create dummy input
            dummy_input = pd.DataFrame([{col: 0.0 for col in feature_columns}])
            prediction = model.predict(dummy_input)
            print(f"✅ Prediction successful: {prediction[0]:.2f}")
        except Exception as e:
            print(f"❌ Prediction failed: {e}")
            return False
        
        print("🎉 Model test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def main():
    """Test all available models."""
    print("🚀 Model Loading Test Suite")
    print("=" * 50)
    
    models_path = Path(MODELS_DIR)
    if not models_path.exists():
        print(f"❌ Models directory not found: {models_path}")
        return
    
    model_files = list(models_path.glob("*.pkl"))
    if not model_files:
        print("❌ No model files found")
        return
    
    print(f"📁 Found {len(model_files)} model files:")
    for f in model_files:
        print(f"   - {f.name}")
    
    # Test each model
    results = {}
    for model_file in model_files:
        results[model_file.name] = test_model_loading(model_file.name)
    
    # Summary
    print("\n📊 SUMMARY")
    print("=" * 50)
    for model_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {model_name}")
    
    passed = sum(results.values())
    total = len(results)
    print(f"\n🎯 Overall: {passed}/{total} models passed")


if __name__ == "__main__":
    main()
