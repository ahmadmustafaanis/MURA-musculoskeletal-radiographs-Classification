import tensorflow as tf

backbone = tf.keras.applications.resnet.ResNet50(
    include_top=False, weights="imagenet", input_shape=(256, 256, 3), pooling="avg"
)


def return_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(3, 1, input_shape=(256, 256, 1)))
    model.add(backbone)
    model.add(tf.keras.layers.Dense(128, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

    model.layers[1].trainable = False  # resnet50 will not be trained

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.MeanSquaredError(),
            tf.keras.metrics.RootMeanSquaredError(),
        ],
    )

    return model
