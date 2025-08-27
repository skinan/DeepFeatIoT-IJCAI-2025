import pandas as pd

def load_data(dataset_name):
    if dataset_name == "Swiss":
        drop_columns = [
            "database_id",
            "metadata_0",
            "metadata_1",
            "metadata_2",
            "metadata_3",
            "metadata_4",
            "metadata_5",
            "metadata_6",
            "label",
        ]
        df = pd.read_csv(
            "../datasets/Swiss.csv"
        )
        X = df.drop(labels=drop_columns, axis=1)

    elif dataset_name == "Urban":
        drop_columns = [
            "database_id",
            "metadata_0",
            "metadata_1",
            "metadata_2",
            "metadata_3",
            "metadata_4",
            "metadata_5",
            "metadata_6",
            "label",
        ]
        df = pd.read_csv(
            "../datasets/Urban.csv"
        )
        X = df.drop(labels=drop_columns, axis=1)

    elif dataset_name == "IOWA":
        drop_columns = [
            "label",
        ]
        df = pd.read_csv(
            "../datasets/Iowa.csv"
        )
        X = df.drop(labels=drop_columns, axis=1)

    elif dataset_name == "BAS":
        label_dict = {"co2": 0, "humidity": 1, "light": 2, "pir": 3, "temperature": 4}

        df = pd.read_csv(
            "../datasets/SBAS.csv",
        )
        df["label"] = df["label"].replace(label_dict)
        X = df.drop(labels=["label"], axis=1)
        mask_value = -1
        X = X.fillna(mask_value)

    else:
        raise KeyError(
            "Wrong Dataset Name Provided / Selected Dataset Option Unavailable"
        )

    X = X.astype(float)
    y = df["label"]

    return X, y


if __name__ == "__main__":
    print("This is a module")
