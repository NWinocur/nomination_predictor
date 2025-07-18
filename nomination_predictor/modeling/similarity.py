"""
Model-aware similarity functions for finding similar historical cases.

This module provides functions to find the most similar historical instances
to a given input, using the trained model's feature importances to weight
the similarity calculation.
"""

from typing import Any, Dict, Optional, Union

from loguru import logger
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor


def get_feature_importances(model: Union[Pipeline, XGBRegressor]) -> np.ndarray:
    """
    Extract feature importances from a trained model.

    Parameters
    ----------
    model : Pipeline or XGBRegressor
        Trained model to extract importances from

    Returns
    -------
    np.ndarray
        Feature importances array
    """
    if isinstance(model, Pipeline):
        # Get the final estimator from the pipeline
        estimator = model[-1]
    else:
        estimator = model

    if hasattr(estimator, "feature_importances_"):
        return estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        # For linear models, use absolute coefficients
        return np.abs(estimator.coef_)
    else:
        logger.warning("Model does not have feature importances, using uniform weights")
        # Fallback: assume all features are equally important
        return (
            np.ones(len(model.feature_names_in_)) if hasattr(model, "feature_names_in_") else None
        )


def weighted_similarity(
    query_row: np.ndarray,
    reference_data: np.ndarray,
    feature_weights: np.ndarray,
    metric: str = "cosine",
) -> np.ndarray:
    """
    Calculate weighted similarity between a query row and reference data.

    Parameters
    ----------
    query_row : np.ndarray
        Single row to find similarities for (1D array)
    reference_data : np.ndarray
        Reference dataset (2D array, rows are samples)
    feature_weights : np.ndarray
        Weights for each feature (from model importance)
    metric : str, default="cosine"
        Similarity metric: "cosine", "euclidean", or "manhattan"

    Returns
    -------
    np.ndarray
        Similarity scores for each row in reference_data
    """
    # Apply feature weights by scaling both query and reference data
    weighted_query = query_row * np.sqrt(feature_weights)
    weighted_reference = reference_data * np.sqrt(feature_weights)

    if metric == "cosine":
        # Cosine similarity (higher = more similar)
        similarities = cosine_similarity(
            weighted_query.reshape(1, -1), weighted_reference
        ).flatten()
        return similarities

    elif metric == "euclidean":
        # Euclidean distance (lower = more similar, so we invert)
        distances = euclidean_distances(
            weighted_query.reshape(1, -1), weighted_reference
        ).flatten()
        # Convert to similarity (higher = more similar)
        max_dist = np.max(distances)
        similarities = 1 - (distances / max_dist) if max_dist > 0 else np.ones_like(distances)
        return similarities

    elif metric == "manhattan":
        # Manhattan distance
        distances = np.sum(np.abs(weighted_reference - weighted_query), axis=1)
        max_dist = np.max(distances)
        similarities = 1 - (distances / max_dist) if max_dist > 0 else np.ones_like(distances)
        return similarities

    else:
        raise ValueError(f"Unsupported metric: {metric}")


def find_most_similar_case(
    query_row: Union[pd.Series, np.ndarray, Dict[str, Any]],
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    model: Union[Pipeline, XGBRegressor],
    feature_columns: Optional[list] = None,
    metric: str = "cosine",
    return_top_k: int = 1,
    include_similarity_score: bool = True,
) -> Union[pd.Series, pd.DataFrame]:
    """
    Find the most similar historical case(s) to a query using model-aware similarity.

    Parameters
    ----------
    query_row : pd.Series, np.ndarray, or dict
        The input case to find similarities for
    train_data : pd.DataFrame
        Training dataset
    test_data : pd.DataFrame
        Test dataset
    model : Pipeline or XGBRegressor
        Trained model to extract feature importances from
    feature_columns : list, optional
        List of feature column names. If None, inferred from model
    metric : str, default="cosine"
        Similarity metric to use
    return_top_k : int, default=1
        Number of most similar cases to return
    include_similarity_score : bool, default=True
        Whether to include similarity scores in output

    Returns
    -------
    pd.Series or pd.DataFrame
        Most similar case(s) with optional similarity scores
    """
    # Combine train and test data
    combined_data = pd.concat([train_data, test_data], ignore_index=True)
    combined_data["_dataset_source"] = ["train"] * len(train_data) + ["test"] * len(test_data)

    # Get feature importances
    feature_weights = get_feature_importances(model)

    if feature_weights is None:
        raise ValueError("Could not extract feature importances from model")

    # Determine feature columns
    if feature_columns is None:
        if hasattr(model, "feature_names_in_"):
            feature_columns = model.feature_names_in_
        else:
            raise ValueError(
                "feature_columns must be provided if model doesn't have feature_names_in_"
            )

    # Ensure query_row is properly formatted
    if isinstance(query_row, dict):
        query_row = pd.Series(query_row)
    elif isinstance(query_row, np.ndarray):
        query_row = pd.Series(query_row, index=feature_columns)

    # Extract feature data
    query_features = query_row[feature_columns].values
    reference_features = combined_data[feature_columns].values

    # Handle missing values (fill with column means)
    if np.any(np.isnan(query_features)):
        col_means = np.nanmean(reference_features, axis=0)
        query_features = np.where(np.isnan(query_features), col_means, query_features)

    if np.any(np.isnan(reference_features)):
        col_means = np.nanmean(reference_features, axis=0)
        reference_features = np.where(np.isnan(reference_features), col_means, reference_features)

    # Calculate similarities
    similarities = weighted_similarity(
        query_features, reference_features, feature_weights, metric=metric
    )

    # Get top-k most similar cases
    top_indices = np.argsort(similarities)[-return_top_k:][::-1]  # Descending order

    result_data = combined_data.iloc[top_indices].copy()

    if include_similarity_score:
        result_data["similarity_score"] = similarities[top_indices]

    # Add explanation of why this case is similar
    if return_top_k == 1:
        result_data["_explanation"] = _generate_similarity_explanation(
            query_row, result_data.iloc[0], feature_columns, feature_weights
        )
        return result_data.iloc[0]
    else:
        for i in range(len(result_data)):
            result_data.iloc[i, result_data.columns.get_loc("_explanation")] = (
                _generate_similarity_explanation(
                    query_row, result_data.iloc[i], feature_columns, feature_weights
                )
            )
        return result_data


def _generate_similarity_explanation(
    query_row: pd.Series,
    similar_row: pd.Series,
    feature_columns: list,
    feature_weights: np.ndarray,
    top_features: int = 3,
) -> str:
    """
    Generate a human-readable explanation of why two cases are similar.

    Parameters
    ----------
    query_row : pd.Series
        The query case
    similar_row : pd.Series
        The similar case found
    feature_columns : list
        List of feature column names
    feature_weights : np.ndarray
        Feature importance weights
    top_features : int, default=3
        Number of top contributing features to mention

    Returns
    -------
    str
        Human-readable explanation
    """
    # Calculate feature-wise similarities weighted by importance
    feature_similarities = []

    for i, feature in enumerate(feature_columns):
        query_val = query_row[feature]
        similar_val = similar_row[feature]

        # Calculate similarity for this feature (1 - normalized absolute difference)
        if pd.isna(query_val) or pd.isna(similar_val):
            feature_sim = 0.5  # Neutral similarity for missing values
        else:
            # Normalize by the range of values for this feature (rough approximation)
            feature_range = abs(query_val) + abs(similar_val) + 1e-8  # Avoid division by zero
            feature_sim = 1 - abs(query_val - similar_val) / feature_range

        # Weight by feature importance
        weighted_sim = feature_sim * feature_weights[i]
        feature_similarities.append((feature, weighted_sim, query_val, similar_val))

    # Sort by weighted similarity contribution
    feature_similarities.sort(key=lambda x: x[1], reverse=True)

    # Generate explanation for top features
    explanations = []
    for feature, _, query_val, similar_val in feature_similarities[:top_features]:
        if pd.isna(query_val) or pd.isna(similar_val):
            explanations.append(f"{feature}: missing data")
        else:
            explanations.append(f"{feature}: {query_val:.2f} vs {similar_val:.2f}")

    return f"Most similar based on: {'; '.join(explanations)}"


def batch_find_similar_cases(
    query_data: pd.DataFrame,
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    model: Union[Pipeline, XGBRegressor],
    feature_columns: Optional[list] = None,
    metric: str = "cosine",
) -> pd.DataFrame:
    """
    Find most similar cases for multiple queries at once.

    Parameters
    ----------
    query_data : pd.DataFrame
        Multiple cases to find similarities for
    train_data : pd.DataFrame
        Training dataset
    test_data : pd.DataFrame
        Test dataset
    model : Pipeline or XGBRegressor
        Trained model
    feature_columns : list, optional
        Feature column names
    metric : str, default="cosine"
        Similarity metric

    Returns
    -------
    pd.DataFrame
        Results with similar cases for each query
    """
    results = []

    for idx, query_row in query_data.iterrows():
        similar_case = find_most_similar_case(
            query_row, train_data, test_data, model, feature_columns, metric, return_top_k=1
        )

        result_row = {
            "query_index": idx,
            "similar_case_index": similar_case.name,
            "similarity_score": similar_case.get("similarity_score", np.nan),
            "dataset_source": similar_case.get("_dataset_source", "unknown"),
            "explanation": similar_case.get("_explanation", ""),
        }

        # Add key fields from the similar case
        for col in ["nominee_name", "receiveddate", "days_nom_to_conf"]:  # Adjust as needed
            if col in similar_case:
                result_row[f"similar_{col}"] = similar_case[col]

        results.append(result_row)

    return pd.DataFrame(results)
