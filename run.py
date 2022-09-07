import argparse
import sys

import tensorflow as tf

import utils
from dataloaders import return_dataloaders
from model import return_model

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--model_path", type=str, default="models/anyfilepath.h5")
parser.add_argument("--data_path", type=str, default="../input/mura-v1.1/MURA-v1.1")

sys.stdout = open("training_logs.txt", "w", buffering=1, encoding="utf-8")

train_dataloader, valid_dataloader, test_dataloader = return_dataloaders(
    parser.parse_args().data_path,
    parser.parse_args().batch_size,
)

my_callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        parser.parse_args().model_path, save_best_only=True, monitor="val_loss"
    )  # save best model based on val_loss. we can change it to any metric
]
model = return_model()
history = model.fit(
    train_dataloader,
    validation_data=valid_dataloader,
    epochs=parser.parse_args().epochs,
    callbacks=my_callbacks,
    # batchsize
)

utils.visualize_training(history)
utils.evaluate_model(model, test_dataloader)
