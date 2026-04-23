"""Utilities for task-specific dataset formatting."""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import numpy as np
from typing import Iterable, Optional, Sequence, Union

SUPPORTED_TASK_TYPES = (
    "imputation",
    "forecasting",
    "classification",
    "clustering",
    "anomaly_detection",
)


def _normalize_feature_indices(
    feature_indices: Optional[Union[int, Sequence[int], np.ndarray]],
    n_features: int,
) -> np.ndarray:
    if feature_indices is None:
        return np.arange(n_features)

    if isinstance(feature_indices, int):
        indices = np.asarray([feature_indices])
    elif isinstance(feature_indices, np.ndarray):
        indices = feature_indices
    elif isinstance(feature_indices, Iterable):
        indices = np.asarray(list(feature_indices))
    else:
        raise ValueError("forecast_feature_indices must be None, int, or an iterable of ints.")

    if indices.ndim != 1 or len(indices) == 0:
        raise ValueError("forecast_feature_indices must contain at least one feature index.")

    if np.any(indices < 0) or np.any(indices >= n_features):
        raise ValueError(f"forecast_feature_indices must be in [0, {n_features - 1}], but got {indices.tolist()}.")

    return indices.astype(int)


def convert_processed_dataset_by_task_type(
    processed_dataset: dict,
    task_type: str = "imputation",
    n_pred_steps: int = 1,
    forecast_feature_indices: Optional[Union[int, Sequence[int], np.ndarray]] = None,
) -> dict:
    """Convert a processed dataset dict into task-specific format.

    Parameters
    ----------
    processed_dataset:
        Output dict built by a preprocess function.

    task_type:
        One of ['imputation', 'forecasting', 'classification', 'clustering', 'anomaly_detection'].

    n_pred_steps:
        Forecasting horizon used when ``task_type='forecasting'``.

    forecast_feature_indices:
        Feature indices used as forecasting targets. If None, all features are used.

    Returns
    -------
    dict
        The task-formatted processed dataset dictionary.
    """
    if task_type not in SUPPORTED_TASK_TYPES:
        raise ValueError(f"task_type must be one of {SUPPORTED_TASK_TYPES}, but got {task_type}.")

    processed_dataset["task_type"] = task_type

    if task_type != "forecasting":
        # so far other tasks do not need further processing steps
        return processed_dataset

    if not isinstance(n_pred_steps, int) or n_pred_steps <= 0:
        raise ValueError(f"n_pred_steps must be a positive integer, but got {n_pred_steps}.")

    train_X = processed_dataset["train_X"]
    n_steps = train_X.shape[1]
    n_features = train_X.shape[2]
    if n_pred_steps >= n_steps:
        raise ValueError(f"n_pred_steps ({n_pred_steps}) must be smaller than n_steps ({n_steps}).")

    target_indices = _normalize_feature_indices(forecast_feature_indices, n_features)

    for split in ["train", "val", "test"]:
        X_key = f"{split}_X"
        X_ori_key = f"{split}_X_ori"
        X = processed_dataset[X_key]

        processed_dataset[f"{split}_X_pred"] = X[:, -n_pred_steps:, target_indices]
        processed_dataset[X_key] = X[:, :-n_pred_steps, :]

        if X_ori_key in processed_dataset:
            X_ori = processed_dataset[X_ori_key]
            processed_dataset[X_ori_key] = X_ori[:, :-n_pred_steps, :]
            # if the original data without missingness is provided,
            # we replace *_X_pred with the original values for forecasting targets,
            # which can be used for evaluation of forecasting performance without influence of artificial missingness
            processed_dataset[f"{split}_X_pred"] = X_ori[:, -n_pred_steps:, target_indices]

    processed_dataset["n_pred_steps"] = n_pred_steps
    processed_dataset["n_pred_features"] = len(target_indices)
    processed_dataset["n_steps_original"] = n_steps
    processed_dataset["n_steps"] = n_steps - n_pred_steps

    return processed_dataset
