"""
Preprocessing func for the UCR&UAE datasets.

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import tsdb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ..utils.logging import logger, print_final_dataset_info
from ..utils.missingness import create_missingness


def preprocess_ucr_uea_datasets(
    dataset_name,
    rate,
    pattern: str = "point",
    **kwargs,
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

    Returns
    -------
    processed_dataset :
        A dictionary containing the processed UCR&UEA dataset.

    """

    assert 0 <= rate < 1, f"rate must be in [0, 1), but got {rate}"
    assert dataset_name.startswith(
        "ucr_uea_"
    ), f"set_name must start with 'ucr_uea_', but got {dataset_name}"
    assert dataset_name in tsdb.list(), f"{dataset_name} is not in TSDB database."

    data = tsdb.load(dataset_name)
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    n_X_train = len(X_train)

    train_ids, val_ids = train_test_split(list(range(n_X_train)), test_size=0.2)
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

    if rate > 0:
        # hold out ground truth in the original data for evaluation
        train_X_ori = train_X
        val_X_ori = val_X
        test_X_ori = test_X

        # mask values in the train set to keep the same with below validation and test sets
        train_X = create_missingness(train_X, rate, pattern, **kwargs)
        # mask values in the validation set as ground truth
        val_X = create_missingness(val_X, rate, pattern, **kwargs)
        # mask values in the test set as ground truth
        test_X = create_missingness(test_X, rate, pattern, **kwargs)

        processed_dataset["train_X"] = train_X
        processed_dataset["train_X_ori"] = train_X_ori

        processed_dataset["val_X"] = val_X
        processed_dataset["val_X_ori"] = val_X_ori

        processed_dataset["test_X"] = test_X
        # test_X_ori is for error calc, not for model input, hence mustn't have NaNs
        processed_dataset["test_X_ori"] = test_X_ori
    else:
        logger.warning("rate is 0, no missing values are artificially added.")

    print_final_dataset_info(train_X, val_X, test_X)
    return processed_dataset
