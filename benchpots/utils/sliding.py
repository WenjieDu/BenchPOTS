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
    window_len: int,
    sliding_len: int = None,
) -> Union[np.ndarray, torch.Tensor]:
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

    window_len :
        The length of the sliding window, i.e. the number of time steps in the generated data samples.

    sliding_len :
        The sliding length of the window for each moving step. It will be set as the same with n_steps if None.

    Returns
    -------
    samples :
        The generated time-series data samples of shape [seq_len//sliding_len, n_steps, n_features].

    """
    sliding_len = window_len if sliding_len is None else sliding_len
    total_len = time_series.shape[0]
    start_indices = np.asarray(range(total_len // sliding_len)) * sliding_len

    # remove the last one if left length is not enough
    if total_len - start_indices[-1] < window_len:
        left_len = total_len - start_indices[-1]
        to_drop = math.floor(window_len / sliding_len)
        logger.warning(
            f"{total_len}-{start_indices[-1]}={left_len} < {window_len}. "
            f"The last {to_drop} samples are dropped due to the left length {left_len} is not enough."
        )
        start_indices = start_indices[:-to_drop]

    sample_collector = []
    for idx in start_indices:
        sample_collector.append(time_series[idx : idx + window_len])

    if isinstance(time_series, torch.Tensor):
        samples = torch.cat(sample_collector, dim=0)
    elif isinstance(time_series, np.ndarray):
        samples = np.asarray(sample_collector).astype("float32")
    else:
        raise RuntimeError

    return samples


def inverse_sliding_window(X, sliding_len):
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

    sliding_len :
        The sliding length of the window for each moving step in the sliding window method used to generate X.

    Returns
    -------
    restored_data :
        The restored time-series data with shape of [total_length, n_features].

    """
    assert len(X.shape) == 3, f"X should be a 3D array, but got {X.shape}"
    n_samples, window_size, n_features = X.shape

    if sliding_len >= window_size:
        if sliding_len > window_size:
            logger.warning(
                f"sliding_len {sliding_len} is larger than the window size {window_size}, "
                f"hence there will be gaps between restored data."
            )
        restored_data = X.reshape(n_samples * window_size, n_features)
    else:
        collector = [X[0][:sliding_len]]
        overlap = X[0][sliding_len:]
        for x in X[1:]:
            overlap_avg = (overlap + x[:-sliding_len]) / 2
            collector.append(overlap_avg[:sliding_len])
            overlap = np.concatenate(
                [overlap_avg[sliding_len:], x[-sliding_len:]], axis=0
            )
        collector.append(overlap)
        restored_data = np.concatenate(collector, axis=0)
    return restored_data
