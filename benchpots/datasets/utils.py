"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


from pygrinder import calc_missing_rate, mcar, seq_missing, block_missing
from pypots.utils.logging import logger


def create_missingness(X, rate, pattern, **kwargs):
    """Create missingness in the data.

    Parameters
    ----------
    X:
        The input data.

    rate:
        The missing rate.

    pattern:
        The missing pattern to apply to the dataset.
        Must be one of ['point', 'subseq', 'block'].

    Returns
    -------

    """
    supported_missing_pattern = ["point", "subseq", "block"]

    assert 0 < rate < 1, "rate must be in [0, 1)"
    assert (
        pattern.lower() in supported_missing_pattern
    ), f"pattern must be one of {supported_missing_pattern}, but got {pattern}"

    if pattern == "point":
        return mcar(X, rate)
    elif pattern == "subseq":
        return seq_missing(X, rate, **kwargs)
    elif pattern == "block":
        return block_missing(X, factor=rate, **kwargs)
    else:
        raise ValueError(f"Unknown missingness pattern: {pattern}")


def print_final_dataset_info(train_X, val_X, test_X):
    train_set_size, val_set_size, test_set_size = len(train_X), len(val_X), len(test_X)
    total_size = len(train_X) + len(val_X) + len(test_X)
    n_steps, n_features = train_X.shape[1], train_X.shape[2]

    logger.info(f"Total sample number: {total_size}")
    logger.info(
        f"Training set size: {train_set_size} ({train_set_size / total_size:.2%})"
    )
    logger.info(
        f"Validation set size: {val_set_size} ({val_set_size / total_size:.2%})"
    )
    logger.info(f"Test set size: {test_set_size} ({test_set_size / total_size:.2%})")
    logger.info(f"Number of steps: {n_steps}")
    logger.info(f"Number of features: {n_features}")
    logger.info(f"Train set missing rate: {calc_missing_rate(train_X):.2%}")
    logger.info(f"Validating set missing rate: {calc_missing_rate(val_X):.2%}")
    logger.info(f"Test set missing rate: {calc_missing_rate(test_X):.2%}")
