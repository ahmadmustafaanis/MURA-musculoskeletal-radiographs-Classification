import os

import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from preprocessor import find_imp_area_xray

TRAIN_CSV_NAME = "train_image_paths.csv"
VALID_CSV_NAME = "valid_image_paths.csv"


def data_loader(csv_path):
    df = pd.read_csv(csv_path, dtype="str", header=None)

    df.columns = ["path"]
    df["labels"] = df["path"].map(
        lambda x: "positive" if "positive" in x else "negative"
    )
    df["path"] = df["path"].map(lambda x: x.replace("MURA-v1.1/", ""))
    # print(df["path"])
    return df


def return_dataloaders(PATH, batch_size):

    df_train = data_loader(os.path.join(PATH, TRAIN_CSV_NAME))
    df_test = data_loader(os.path.join(PATH, VALID_CSV_NAME))

    df_train["sub_class"] = df_train["path"].str.split("/").map(lambda x: x[2])
    df_test["sub_class"] = df_test["path"].str.split("/").map(lambda x: x[2])
    # print(df_train.head().iloc[0])
    # DOPPING ROWS OF FOREARM AND HUMERUS
    df_train.drop(
        df_train[
            (df_train["sub_class"] == "XR_HUMERUS")
            | (df_train["sub_class"] == "XR_FOREARM")
        ].index,
        inplace=True,
    )

    df_test.drop(
        df_test[
            (df_test["sub_class"] == "XR_HUMERUS")
            | (df_test["sub_class"] == "XR_FOREARM")
        ].index,
        inplace=True,
    )

    X_train, X_valid, y_train, y_valid = train_test_split(
        df_train[["path"]], df_train[["labels"]]
    )

    df_train = pd.concat((X_train, y_train), axis=1)
    df_valid = pd.concat((X_valid, y_valid), axis=1)

    train_generator = ImageDataGenerator(
        rotation_range=45,  # 90 degree random rotation
        horizontal_flip=True,
        vertical_flip=True,
        preprocessing_function=find_imp_area_xray,
    )

    valid_generator = ImageDataGenerator(preprocessing_function=find_imp_area_xray)

    train_generator_flow = train_generator.flow_from_dataframe(
        dataframe=df_train,
        x_col="path",
        y_col="labels",
        directory=PATH,
        class_mode="binary",
        classes=["positive", "negative"],
        subset=None,
        shuffle=True,
        batch_size=batch_size,
        color_mode="grayscale",
    )

    valid_generator_flow = valid_generator.flow_from_dataframe(
        dataframe=df_valid,
        x_col="path",
        y_col="labels",
        directory=PATH,
        class_mode="binary",
        classes=["positive", "negative"],
        subset=None,
        shuffle=False,
        batch_size=batch_size,
        color_mode="grayscale",
    )

    test_generator_flow = valid_generator.flow_from_dataframe(
        dataframe=df_test,
        x_col="path",
        y_col="labels",
        directory=PATH,
        class_mode="binary",
        classes=["positive", "negative"],
        subset=None,
        shuffle=True,
        seed=3407,
        batch_size=batch_size,
        color_mode="grayscale",
    )

    return train_generator_flow, valid_generator_flow, test_generator_flow
