"""
Configure logging here.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from pygrinder import calc_missing_rate
from tsdb.utils.logging import Logger
from typing import Any

# initialize a logger for PyPOTS logging
logger_creator = Logger(name="BenchPOTS running log")
logger = logger_creator.logger


def print_final_dataset_info(
    processed_dataset: dict,
) -> None:
    train_X = processed_dataset["train_X"]
    val_X = processed_dataset["val_X"]
    test_X = processed_dataset["test_X"]

    train_set_size, val_set_size, test_set_size = len(train_X), len(val_X), len(test_X)
    total_size = len(train_X) + len(val_X) + len(test_X)
    n_steps, n_features = train_X.shape[1], train_X.shape[2]

    logger.info(f"Total sample number: {total_size}")
    logger.info(f"Training set size: {train_set_size} ({train_set_size / total_size:.2%})")
    logger.info(f"Validation set size: {val_set_size} ({val_set_size / total_size:.2%})")
    logger.info(f"Test set size: {test_set_size} ({test_set_size / total_size:.2%})")
    logger.info(f"Number of steps: {n_steps}")
    logger.info(f"Number of features: {n_features}")
    logger.info(f"Train set missing rate: {calc_missing_rate(train_X):.2%}")
    logger.info(f"Validating set missing rate: {calc_missing_rate(val_X):.2%}")
    logger.info(f"Test set missing rate: {calc_missing_rate(test_X):.2%}")

    if processed_dataset.get("task_type", "imputation") == "forecasting":
        logger.info("Task type: forecasting")
        if processed_dataset.get("n_steps_original") is not None:
            logger.info(f"Original number of steps: {processed_dataset['n_steps_original']}")
        if processed_dataset.get("n_pred_steps") is not None:
            logger.info(f"Number of prediction steps: {processed_dataset['n_pred_steps']}")
        if processed_dataset.get("n_pred_features") is not None:
            logger.info(f"Number of prediction features: {processed_dataset['n_pred_features']}")

        if processed_dataset.get("train_X_pred") is not None:
            logger.info(
                f"Train prediction target missing rate: {calc_missing_rate(processed_dataset['train_X_pred']):.2%}"
            )
        if processed_dataset.get("val_X_pred") is not None:
            logger.info(
                f"Validation prediction target missing rate: {calc_missing_rate(processed_dataset['val_X_pred']):.2%}"
            )
        if processed_dataset.get("test_X_pred") is not None:
            logger.info(
                f"Test prediction target missing rate: {calc_missing_rate(processed_dataset['test_X_pred']):.2%}"
            )
