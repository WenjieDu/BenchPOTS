"""
Utils about sliding window method for time series data.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import math
from typing import Union

import numpy as np
import torch

from ..utils.logging import logger


def sliding_window(
    time_series: Union[np.ndarray, torch.Tensor],
    window_size: int,
    stride: int = None,
    drop_last: bool = True,
) -> Union[np.ndarray, torch.Tensor, tuple]:
    """Generate time series samples with sliding window method, truncating windows from time-series data
    with a given sequence length.

    Given a time series of shape [seq_len, n_features] (seq_len is the total sequence length of the time series), this
    sliding_window function will generate time-series samples from this given time series with sliding window method.
    The number of generated samples is seq_len//sliding_len. And the final returned numpy ndarray has a shape
    [seq_len//sliding_len, n_steps, n_features].

    Parameters
    ----------
    time_series :
        time series data, len(shape)=2, [total_length, feature_num]

    window_size :
        Former window_len. The size of the window, i.e. the number of time steps in the generated data samples.

    stride :
        Former sliding_len. The stride, i.e. the sliding length, of the window for each moving step.

    drop_last :
        Whether to drop the last sample if the left length is not enough for a complete window.

    Returns
    -------
    samples :
        The generated time-series data samples of shape [seq_len//sliding_len, n_steps, n_features].

    """
    stride = window_size if stride is None else stride

    # check input
    assert len(time_series) > window_size, (
        f"time_series length {len(time_series)} is less than "
        f"window_size {window_size}. There is no space for sliding."
    )
    assert (
        stride > 0 and window_size > 0
    ), f"stride {stride} and window_size {window_size} must be positive"
    assert stride <= window_size, (
        f"stride {stride} shouldn't be larger than window_size {window_size}. "
        f"Otherwise there will be gaps between samples."
    )

    total_len = time_series.shape[0]
    start_indices = np.asarray(range(total_len // stride)) * stride

    # remove the last samples if left length is not enough
    if total_len - start_indices[-1] < window_size:
        left_len = total_len - start_indices[-1]
        to_drop = math.floor(window_size / stride)
        logger.warning(
            f"{total_len}-{start_indices[-1]}={left_len} < {window_size}. "
            f"{to_drop} incomplete samples are dropped due to the left length {left_len} is not enough."
        )
        start_indices = start_indices[:-to_drop]

    sample_collector = []
    for idx in start_indices:
        sample_collector.append(time_series[idx : idx + window_size])

    if isinstance(time_series, torch.Tensor):
        samples = torch.cat(sample_collector, dim=0)
    elif isinstance(time_series, np.ndarray):
        samples = np.asarray(sample_collector).astype("float32")
    else:
        raise RuntimeError

    if not drop_last:
        logger.info(
            "drop_last is set as False, the last sample is kept and will be returned independently."
        )
        return samples, time_series[start_indices[-1] + stride :]

    return samples


def inverse_sliding_window(X, stride):
    """Restore the original time-series data from the generated sliding window samples.
    Note that this is the inverse operation of the `sliding_window` function, but there is no guarantee that
    the restored data is the same as the original data considering that
    1. the sliding length may be larger than the window size and there will be gaps between restored data;
    2. if values in the samples get changed, the overlap part may not be the same as the original data after averaging;
    3. some incomplete samples at the tail may be dropped during the sliding window operation, hence the restored data
       may be shorter than the original data.

    Parameters
    ----------
    X :
        The generated time-series samples with sliding window method, shape of [n_samples, n_steps, n_features],
        where n_steps is the window size of the used sliding window method.

    stride :
        The stride (sliding length) of the window for each moving step in the sliding window method used to generate X.

    Returns
    -------
    restored_data :
        The restored time-series data with shape of [total_length, n_features].

    """
    assert len(X.shape) == 3, f"X should be a 3D array, but got {X.shape}"
    n_samples, window_size, n_features = X.shape

    if stride >= window_size:
        if stride > window_size:
            logger.warning(
                f"sliding_len {stride} is larger than the window size {window_size}, "
                f"hence there will be gaps between restored data."
            )
        restored_data = X.reshape(n_samples * window_size, n_features)
    else:
        collector = [X[0][:stride]]
        overlap = X[0][stride:]
        for x in X[1:]:
            overlap_avg = (overlap + x[:-stride]) / 2
            collector.append(overlap_avg[:stride])
            overlap = np.concatenate([overlap_avg[stride:], x[-stride:]], axis=0)
        collector.append(overlap)
        restored_data = np.concatenate(collector, axis=0)
    return restored_data
