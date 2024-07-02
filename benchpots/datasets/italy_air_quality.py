"""
Preprocessing func for the dataset Italy Air Quality.

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import tsdb
from sklearn.preprocessing import StandardScaler

from ..utils.logging import logger, print_final_dataset_info
from ..utils.missingness import create_missingness
from ..utils.sliding import sliding_window


def preprocess_italy_air_quality(
    rate,
    n_steps,
    pattern: str = "point",
    **kwargs,
) -> dict:
    """Load and preprocess the dataset Italy Air Quality.

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

    Returns
    -------
    processed_dataset :
        A dictionary containing the processed Italy Air Quality.
    """

    assert n_steps > 0, f"sample_n_steps must be larger than 0, but got {n_steps}"

    data = tsdb.load("italy_air_quality")
    df = data["X"]
    df = df.drop(columns=["Date", "Time"])
    features = df.columns
    df = df.to_numpy()

    # split the dataset into train, validation, and test sets
    all_n_steps = len(df)
    val_test_len = round(all_n_steps * 0.2)
    train_set = df[: -2 * val_test_len]
    val_set = df[-2 * val_test_len : -val_test_len]
    test_set = df[-val_test_len:]

    scaler = StandardScaler()
    train_set_X = scaler.fit_transform(train_set)
    val_set_X = scaler.transform(val_set)
    test_set_X = scaler.transform(test_set)

    train_X = sliding_window(train_set_X, n_steps)
    val_X = sliding_window(val_set_X, n_steps)
    test_X = sliding_window(test_set_X, n_steps)

    # assemble the final processed data into a dictionary
    processed_dataset = {
        # general info
        "n_steps": n_steps,
        "n_features": len(features),
        "scaler": scaler,
        # train set
        "train_X": train_X,
        # val set
        "val_X": val_X,
        # test set
        "test_X": test_X,
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
