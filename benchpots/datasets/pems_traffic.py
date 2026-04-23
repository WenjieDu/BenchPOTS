"""
Preprocessing func for the dataset PeMS traffic.

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from typing import Any, Optional, Sequence, Union

import pandas as pd
import tsdb
from sklearn.preprocessing import StandardScaler

from ..utils.logging import logger, print_final_dataset_info
from ..utils.missingness import create_missingness
from ..utils.sliding import sliding_window
from ..utils.task_type import convert_processed_dataset_by_task_type


def preprocess_pems_traffic(
    rate: float,
    n_steps: int,
    pattern: str = "point",
    random_state: Optional[int] = None,
    task_type: str = "imputation",
    n_pred_steps: int = 1,
    forecast_feature_indices: Optional[Union[int, Sequence[int]]] = None,
    **kwargs: Any,
) -> dict:
    """Load and preprocess the dataset PeMS traffic.

    Parameters
    ----------
    rate:
        The missing rate.

    n_steps:
        The number of time steps to in the generated data samples.
        Also the window size of the sliding window.

    pattern:
        The missing pattern to apply to the dataset.
        Must be one of ['point', 'subseq', 'block'].

    random_state:
        Controls the randomness of missingness generation.
        Pass an int for reproducible missingness masks across runs.

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
        A dictionary containing the processed PeMS traffic.
    """

    assert 0 <= rate < 1, f"rate must be in [0, 1), but got {rate}"
    assert n_steps > 0, f"sample_n_steps must be larger than 0, but got {n_steps}"

    # read the raw data
    data = tsdb.load("pems_traffic")
    df = data["X"]

    feature_names = df.columns.tolist()
    feature_names.remove("date")
    df["date"] = pd.to_datetime(df["date"])

    unique_months = df["date"].dt.to_period("M").unique()
    selected_as_train = unique_months[:15]  # use the first 15 months as train set
    logger.info(f"months selected as train set are {selected_as_train}")
    selected_as_val = unique_months[15:19]  # select the following 4 months as val set
    logger.info(f"months selected as val set are {selected_as_val}")
    selected_as_test = unique_months[
        19:
    ]  # select the left 6 months as test set, 2018-07 has only 2 days, so can be rounded to 5 months
    logger.info(f"months selected as test set are {selected_as_test}")

    test_set = df[df["date"].dt.to_period("M").isin(selected_as_test)]
    val_set = df[df["date"].dt.to_period("M").isin(selected_as_val)]
    train_set = df[df["date"].dt.to_period("M").isin(selected_as_train)]

    scaler = StandardScaler()
    train_X = scaler.fit_transform(train_set.loc[:, feature_names])
    val_X = scaler.transform(val_set.loc[:, feature_names])
    test_X = scaler.transform(test_set.loc[:, feature_names])

    train_X = sliding_window(train_X, n_steps)
    val_X = sliding_window(val_X, n_steps)
    test_X = sliding_window(test_X, n_steps)

    # assemble the final processed data into a dictionary
    processed_dataset = {
        # general info
        "n_steps": n_steps,
        "n_features": train_X.shape[-1],
        "scaler": scaler,
        # train set
        "train_X": train_X,
        # val set
        "val_X": val_X,
        # test set
        "test_X": test_X,
    }

    processed_dataset = convert_processed_dataset_by_task_type(
        processed_dataset,
        task_type=task_type,
        n_pred_steps=n_pred_steps,
        forecast_feature_indices=forecast_feature_indices,
    )

    if rate > 0:
        if random_state is not None and "random_state" not in kwargs:
            kwargs["random_state"] = random_state

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
