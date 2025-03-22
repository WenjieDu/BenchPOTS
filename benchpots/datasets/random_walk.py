"""
Preprocessing func for the generated random walk dataset.

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


import math
from typing import Optional, Tuple

import numpy as np
from pygrinder import mcar
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

from ..utils.logging import logger, print_final_dataset_info
from ..utils.missingness import create_missingness


def gene_complete_random_walk(
    n_samples: int = 1000,
    n_steps: int = 24,
    n_features: int = 10,
    mu: float = 0.0,
    std: float = 1.0,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """Generate complete random walk time-series data, i.e. having no missing values.

    Parameters
    ----------
    n_samples : int, default=1000
        The number of training time-series samples to generate.

    n_steps: int, default=24
        The number of time steps (length) of generated time-series samples.

    n_features : int, default=10
        The number of features (dimensions) of generated time-series samples.

    mu : float, default=0.0
        Mean of the normal distribution, which random walk steps are sampled from.

    std : float, default=1.0
        Standard deviation of the normal distribution, which random walk steps are sampled from.

    random_state : int, default=None
        Random seed for data generation.

    Returns
    -------
    ts_samples: array, shape of [n_samples, n_steps, n_features]
        Generated random walk time series.
    """
    seed = check_random_state(random_state)
    ts_samples = np.zeros([n_samples, n_steps, n_features])
    random_values = seed.randn(n_samples, n_steps, n_features) * std + mu
    ts_samples[:, 0, :] = random_values[:, 0, :]
    for t in range(1, n_steps):
        ts_samples[:, t, :] = ts_samples[:, t - 1, :] + random_values[:, t, :]
    ts_samples = np.asarray(ts_samples)
    return ts_samples


def gene_complete_random_walk_with_anomalies(
    n_samples: int = 1000,
    n_steps: int = 24,
    n_features: int = 10,
    mu: float = 0.0,
    std: float = 1.0,
    anomaly_rate: float = 0.1,
    anomaly_scale_factor: float = 2.0,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate random walk time-series data for the anomaly-detection task.

    Parameters
    ----------
    n_samples : int, default=1000
        The number of training time-series samples to generate.

    n_features : int, default=10
        The number of features (dimensions) of generated time-series samples.

    n_steps: int, default=24
        The number of time steps (length) of generated time-series samples.

    mu : float, default=0.0
        Mean of the normal distribution, which random walk steps are sampled from.

    std : float, default=1.0
        Standard deviation of the normal distribution, which random walk steps are sampled from.

    anomaly_rate : float, default=0.1
        Proportion of anomaly samples in all samples.

    anomaly_scale_factor : float, default=2.0
        Scale factor for value scaling to create anomaly points in time series samples.

    random_state : int, default=None
        Random seed for data generation.

    Returns
    -------
    X : array, shape of [n_samples, n_steps, n_features]
        Generated time-series data.

    y : array, shape of [n_samples]
        Labels indicating if time-series samples are anomalies.
    """
    assert 0 < anomaly_rate < 1, f"anomaly_proportion should be >0 and <1, but got {anomaly_rate}"

    seed = check_random_state(random_state)
    n_total_steps = n_samples * n_steps
    X = seed.randn(n_total_steps, n_features) * std + mu
    n_anomaly = math.floor(n_total_steps * anomaly_rate)
    anomaly_indices = np.random.choice(n_total_steps, size=n_anomaly, replace=False)

    flatten_X = X.flatten()
    min_val = flatten_X.min()
    max_val = flatten_X.max()
    max_difference = min_val - max_val
    for a_i in anomaly_indices:
        anomaly_sample = X[a_i]

        # which feature to be anomaly
        feat_idx = np.random.choice(a=n_features, size=1, replace=False)

        anomaly_sample[feat_idx] = mu + np.random.uniform(
            low=min_val - anomaly_scale_factor * max_difference,
            high=max_val + anomaly_scale_factor * max_difference,
        )
        X[a_i] = anomaly_sample

    # create labels
    y = np.zeros(n_total_steps)
    y[anomaly_indices] = 1

    X = X.reshape(n_samples, n_steps, n_features)
    y = y.reshape(n_samples, n_steps, 1)

    # shuffling
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]

    return X, y


def gene_complete_random_walk_for_classification(
    n_classes: int = 2,
    n_samples_each_class: int = 500,
    n_steps: int = 24,
    n_features: int = 10,
    anomaly_rate: float = 0,
    shuffle: bool = True,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate complete random walk time-series data for the classification task.

    Parameters
    ----------
    n_classes : int, must >=1, default=2
        Number of classes (types) of the generated data.

    n_samples_each_class : int, default=500
        Number of samples for each class to generate.

    n_steps : int, default=24
        Number of time steps in each sample.

    n_features : int, default=10
        Number of features.

    anomaly_rate : float, default=0
        Proportion of anomaly samples in all samples.
        Default as 0 means no anomaly samples are generated.

    shuffle : bool, default=True
        Whether to shuffle generated samples.
        If not, you can separate samples of each class according to `n_samples_each_class`.
        For example,
        X_class0=X[:n_samples_each_class],
        X_class1=X[n_samples_each_class:n_samples_each_class*2]

    random_state : int, default=None
        Random seed for data generation.

    Returns
    -------
    X : array, shape of [n_samples, n_steps, n_features]
        Generated time-series data.

    y : array, shape of [n_samples]
        Labels indicating classes of time-series samples.

    """
    assert n_classes > 1, f"n_classes should be >1, but got {n_classes}"
    assert 0 <= anomaly_rate < 1, f"anomaly_rate should be in [0,1), but got {anomaly_rate}"

    ts_collector = []
    label_collector = []
    anomaly_label_collector = []

    mu = 0
    std = 1

    for c_ in range(n_classes):
        if anomaly_rate > 0:
            ts_samples, anomaly_labels = gene_complete_random_walk_with_anomalies(
                n_samples=n_samples_each_class,
                n_steps=n_steps,
                n_features=n_features,
                mu=mu,
                std=std,
                anomaly_rate=anomaly_rate,
                random_state=random_state,
            )
            anomaly_label_collector.extend(anomaly_labels)
        else:
            ts_samples = gene_complete_random_walk(
                n_samples=n_samples_each_class,
                n_steps=n_steps,
                n_features=n_features,
                mu=mu,
                std=std,
                random_state=random_state,
            )

        label_samples = np.asarray([1 for _ in range(n_samples_each_class)]) * c_
        ts_collector.extend(ts_samples)
        label_collector.extend(label_samples)
        mu += 1

    X = np.asarray(ts_collector)
    y = np.asarray(label_collector)
    anomaly_y = np.asarray(anomaly_label_collector)

    # if shuffling, then shuffle the order of samples
    if shuffle:
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]
        anomaly_y = anomaly_y[indices] if len(anomaly_y) > 0 else anomaly_y

    return X, y, anomaly_y


def preprocess_random_walk(
    n_steps=24,
    n_features=10,
    n_classes=2,
    n_samples_each_class=1000,
    anomaly_rate=0,
    missing_rate=0.1,
    pattern: str = "point",
    **kwargs,
) -> dict:
    """Generate a random-walk data.

    Parameters
    ----------
    n_steps : int, default=24
        Number of time steps in each sample.

    n_features : int, default=10
        Number of features.

    n_classes : int, default=2
        Number of classes (types) of the generated data.

    n_samples_each_class : int, default=1000
        Number of samples for each class to generate.

    anomaly_rate : float, default=0
        Proportion of anomaly samples in all samples.
        Default as 0 means no anomaly samples are generated.

    missing_rate : float, default=0.1
        The rate of randomly missing values to generate, should be in [0,1).

    pattern :
        The missing pattern to apply to the dataset.
        Must be one of ['point', 'subseq', 'block'].


    Returns
    -------
    data: dict,
        A dictionary containing the generated data.
    """

    assert 0 <= anomaly_rate < 1, f"anomaly_rate should be in [0,1), but got {anomaly_rate}"
    assert 0 <= missing_rate < 1, f"missing_rate must be in [0,1), but got {missing_rate}"

    # generate samples
    X, y, anomaly_y = gene_complete_random_walk_for_classification(
        n_classes=n_classes,
        n_samples_each_class=n_samples_each_class,
        n_steps=n_steps,
        n_features=n_features,
        anomaly_rate=anomaly_rate,
    )

    # split into train/val/test sets
    if anomaly_rate > 0:
        train_X, test_X, train_y, test_y, train_anomaly_y, test_anomaly_y = train_test_split(
            X, y, anomaly_y, test_size=0.2
        )
        train_X, val_X, train_y, val_y, train_anomaly_y, val_anomaly_y = train_test_split(
            train_X, train_y, train_anomaly_y, test_size=0.2
        )
    else:
        train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2)
        train_X, val_X, train_y, val_y = train_test_split(train_X, train_y, test_size=0.2)

    if missing_rate > 0:
        # create random missing values
        train_X_ori = train_X
        train_X = mcar(train_X, missing_rate)
        # test set is left to mask after normalization

    train_X = train_X.reshape(-1, n_features)
    val_X = val_X.reshape(-1, n_features)
    test_X = test_X.reshape(-1, n_features)
    # normalization
    scaler = StandardScaler()
    train_X = scaler.fit_transform(train_X)
    val_X = scaler.transform(val_X)
    test_X = scaler.transform(test_X)
    # reshape into time series samples
    train_X = train_X.reshape(-1, n_steps, n_features)
    val_X = val_X.reshape(-1, n_steps, n_features)
    test_X = test_X.reshape(-1, n_steps, n_features)
    processed_dataset = {
        # general info
        "n_classes": n_classes,
        "n_steps": n_steps,
        "n_features": n_features,
        "scaler": scaler,
        # train set
        "train_X": train_X,
        "train_y": train_y,
        # val set
        "val_X": val_X,
        "val_y": val_y,
        # test set
        "test_X": test_X,
        "test_y": test_y,
    }

    if anomaly_rate > 0:
        processed_dataset["train_anomaly_y"] = train_anomaly_y
        processed_dataset["val_anomaly_y"] = val_anomaly_y
        processed_dataset["test_anomaly_y"] = test_anomaly_y

    if missing_rate > 0:
        # hold out ground truth in the original data for evaluation
        train_X_ori = scaler.transform(train_X_ori.reshape(-1, n_features)).reshape(-1, n_steps, n_features)
        val_X_ori = val_X
        test_X_ori = test_X

        # mask values in the train set to keep the same with below validation and test sets
        train_X = create_missingness(train_X, missing_rate, pattern, **kwargs)
        # mask values in the validation set as ground truth
        val_X = create_missingness(val_X, missing_rate, pattern, **kwargs)
        # mask values in the test set as ground truth
        test_X = create_missingness(test_X, missing_rate, pattern, **kwargs)

        processed_dataset["train_X"] = train_X
        processed_dataset["train_X_ori"] = train_X_ori

        processed_dataset["val_X"] = val_X
        processed_dataset["val_X_ori"] = val_X_ori

        processed_dataset["test_X"] = test_X
        processed_dataset["test_X_ori"] = test_X_ori
    else:
        logger.warning("rate is 0, no missing values are artificially added.")

    print_final_dataset_info(train_X, val_X, test_X)
    return processed_dataset
