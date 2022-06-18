"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import pandas as pd
import tsdb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def preprocess_physionet2012():
    """Generate a fully-prepared PhysioNet-2012 dataset for model testing.

    Returns
    -------
    data: dict,
        A dictionary containing the generated PhysioNet-2012 dataset.
            X : pandas.DataFrame,
                A dataframe contains all time series vectors from 11988 patients, distinguished by column `RecordID`.
            y : pandas.Series
                The 11988 classification labels of all patients, indicating whether they were deceased.

    """
    data = tsdb.load("physionet_2012")
    data["static_features"].remove("ICUType")  # keep ICUType for now
    # remove the other static features, e.g. age, gender
    X = data["X"].drop(data["static_features"], axis=1)

    def apply_func(df_temp):  # pad and truncate to set the max length of samples as 48
        missing = list(set(range(0, 48)).difference(set(df_temp["Time"])))
        missing_part = pd.DataFrame({"Time": missing})
        df_temp = pd.concat(
            [df_temp, missing_part], ignore_index=False, sort=False
        )  # pad the sample's length to 48 if it doesn't have enough time steps
        df_temp = df_temp.set_index("Time").sort_index().reset_index()
        df_temp = df_temp.iloc[:48]  # truncate
        return df_temp

    X = X.groupby("RecordID").apply(apply_func)
    X = X.drop("RecordID", axis=1)
    X = X.reset_index()
    ICUType = X[["RecordID", "ICUType"]].set_index("RecordID").dropna()
    X = X.drop(["level_1", "ICUType"], axis=1)

    dataset = {
        "X": X,
        "y": data["y"],
        "ICUType": ICUType,
    }

    # generate samples
    X = dataset["X"]
    y = dataset["y"]
    ICUType = dataset["ICUType"]

    all_recordID = X["RecordID"].unique()
    train_set_ids, test_set_ids = train_test_split(all_recordID, test_size=0.2)
    train_set_ids, val_set_ids = train_test_split(train_set_ids, test_size=0.2)
    train_set_ids.sort()
    val_set_ids.sort()
    test_set_ids.sort()
    train_set = X[X["RecordID"].isin(train_set_ids)].sort_values(["RecordID", "Time"])
    val_set = X[X["RecordID"].isin(val_set_ids)].sort_values(["RecordID", "Time"])
    test_set = X[X["RecordID"].isin(test_set_ids)].sort_values(["RecordID", "Time"])

    train_set = train_set.drop(["RecordID", "Time"], axis=1)
    val_set = val_set.drop(["RecordID", "Time"], axis=1)
    test_set = test_set.drop(["RecordID", "Time"], axis=1)
    train_X, val_X, test_X = (
        train_set.to_numpy(),
        val_set.to_numpy(),
        test_set.to_numpy(),
    )

    # normalization
    scaler = StandardScaler()
    train_X = scaler.fit_transform(train_X)
    val_X = scaler.transform(val_X)
    test_X = scaler.transform(test_X)

    # reshape into time series samples
    train_X = train_X.reshape(len(train_set_ids), 48, -1)
    val_X = val_X.reshape(len(val_set_ids), 48, -1)
    test_X = test_X.reshape(len(test_set_ids), 48, -1)

    train_y = y[y.index.isin(train_set_ids)].sort_index()
    val_y = y[y.index.isin(val_set_ids)].sort_index()
    test_y = y[y.index.isin(test_set_ids)].sort_index()
    train_y, val_y, test_y = train_y.to_numpy(), val_y.to_numpy(), test_y.to_numpy()

    train_ICUType = ICUType[ICUType.index.isin(train_set_ids)].sort_index()
    val_ICUType = ICUType[ICUType.index.isin(val_set_ids)].sort_index()
    test_ICUType = ICUType[ICUType.index.isin(test_set_ids)].sort_index()
    train_ICUType, val_ICUType, test_ICUType = (
        train_ICUType.to_numpy(),
        val_ICUType.to_numpy(),
        test_ICUType.to_numpy(),
    )

    data = {
        "n_classes": 2,
        "n_steps": 48,
        "n_features": train_X.shape[-1],
        "train_X": train_X,
        "train_y": train_y.flatten(),
        "train_ICUType": train_ICUType.flatten(),
        "val_X": val_X,
        "val_y": val_y.flatten(),
        "val_ICUType": val_ICUType.flatten(),
        "test_X": test_X,
        "test_y": test_y.flatten(),
        "test_ICUType": test_ICUType.flatten(),
        "scaler": scaler,
    }

    return data
