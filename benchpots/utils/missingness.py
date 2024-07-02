"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


from pygrinder import mcar, seq_missing, block_missing


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
