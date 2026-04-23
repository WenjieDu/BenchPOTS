"""
Preprocessing func for the UCR&UAE datasets.

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from typing import Any, Optional, Sequence, Union

import tsdb
from pandas.api.types import is_string_dtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from ..utils.logging import logger, print_final_dataset_info
from ..utils.missingness import create_missingness
from ..utils.task_type import convert_processed_dataset_by_task_type


def preprocess_ucr_uea_datasets(
    dataset_name: str,
    rate: float,
    pattern: str = "point",
    random_state: Optional[int] = None,
    task_type: str = "imputation",
    n_pred_steps: int = 1,
    forecast_feature_indices: Optional[Union[int, Sequence[int]]] = None,
    **kwargs: Any,
) -> dict:
    """Load and preprocess the dataset from UCR&UEA.

    Parameters
    ----------
    dataset_name:
        The name of the UCR_UEA dataset to be loaded. Must start with 'ucr_uea_'.
        Use tsdb.list() to get all available datasets.

    rate:
        The missing rate.

    pattern:
        The missing pattern to apply to the dataset.
        Must be one of ['point', 'subseq', 'block'].

    random_state:
        Controls the randomness of the train/validation split.
        Pass an int for reproducible splits across runs.

    task_type:
        Task type for postprocessing. Supported values are
        ['imputation', 'forecasting', 'classification', 'clustering', 'anomaly_detection'].

    n_pred_steps:
        Forecasting horizon. Effective only when task_type is 'forecasting'.

    forecast_feature_indices:
        Target feature indices for forecasting labels. If None, all features are used.

    Returns
    -------
    processed_dataset :
        A dictionary containing the processed UCR&UEA dataset.

    """

    assert 0 <= rate < 1, f"rate must be in [0, 1), but got {rate}"
    assert dataset_name.startswith("ucr_uea_"), f"set_name must start with 'ucr_uea_', but got {dataset_name}"
    assert dataset_name in tsdb.list(), f"{dataset_name} is not in TSDB database."

    data = tsdb.load(dataset_name)
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    le = None
    if is_string_dtype(y_train):
        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        y_test = le.transform(y_test)

    n_X_train = len(X_train)

    train_ids, val_ids = train_test_split(list(range(n_X_train)), test_size=0.2, random_state=random_state)
    X_train, X_val = X_train[train_ids], X_train[val_ids]
    y_train, y_val = y_train[train_ids], y_train[val_ids]

    X_train_shape = X_train.shape
    X_val_shape = X_val.shape
    X_test_shape = X_test.shape

    X_train = X_train.reshape(X_train_shape[0], -1)
    X_val = X_val.reshape(X_val_shape[0], -1)
    X_test = X_test.reshape(X_test_shape[0], -1)
    scaler = StandardScaler()
    train_X = scaler.fit_transform(X_train)
    val_X = scaler.transform(X_val)
    test_X = scaler.transform(X_test)

    train_X = train_X.reshape(X_train_shape)
    val_X = val_X.reshape(X_val_shape)
    test_X = test_X.reshape(X_test_shape)

    # assemble the final processed data into a dictionary
    processed_dataset = {
        # general info
        "n_steps": train_X.shape[1],
        "n_features": train_X.shape[-1],
        "scaler": scaler,
        # train set
        "train_X": train_X,
        "train_y": y_train,
        # val set
        "val_X": val_X,
        "val_y": y_val,
        # test set
        "test_X": test_X,
        "test_y": y_test,
    }

    if le is not None:
        processed_dataset["label_encoder"] = le

    processed_dataset = convert_processed_dataset_by_task_type(
        processed_dataset,
        task_type=task_type,
        n_pred_steps=n_pred_steps,
        forecast_feature_indices=forecast_feature_indices,
    )

    if rate > 0:
        # hold out ground truth in the original data for evaluation
        train_X_ori = processed_dataset["train_X"]
        val_X_ori = processed_dataset["val_X"]
        test_X_ori = processed_dataset["test_X"]

        # mask values in the train set to keep the same with below validation and test sets
        train_X = create_missingness(processed_dataset["train_X"], rate, pattern, **kwargs)
        # mask values in the validation set as ground truth
        val_X = create_missingness(processed_dataset["val_X"], rate, pattern, **kwargs)
        # mask values in the test set as ground truth
        test_X = create_missingness(processed_dataset["test_X"], rate, pattern, **kwargs)

        processed_dataset["train_X"] = train_X
        processed_dataset["train_X_ori"] = train_X_ori

        processed_dataset["val_X"] = val_X
        processed_dataset["val_X_ori"] = val_X_ori

        processed_dataset["test_X"] = test_X
        processed_dataset["test_X_ori"] = test_X_ori
    else:
        logger.warning("rate is 0, no missing values are artificially added.")
    print_final_dataset_info(processed_dataset)
    return processed_dataset
