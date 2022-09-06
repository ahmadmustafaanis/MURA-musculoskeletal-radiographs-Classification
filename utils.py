import matplotlib.pyplot as plt
from sklearn.metrics import classification_report


def visualize_training(history):
    plt.plot(history.history["loss"], label="loss")
    plt.plot(history.history["val_loss"], label="val loss")
    plt.legend()
    plt.title("loss vs validation loss")
    plt.savefig("visualizations/loss_vs_val_loss.png")

    plt.plot(history.history["accuracy"], label="accuracy")
    plt.plot(history.history["val_accuracy"], label="val accuracy")
    plt.legend()
    plt.title("acc vs validation acc")
    plt.savefig("visualizations/acc_vs_val_acc.png")


def evaluate_model(model, test_generator_flow):
    predictions = model.predict(test_generator_flow)

    pred = [1 if pred[0] > 0.5 else 0 for pred in predictions]

    print("Classification report:\n---------------")
    print(classification_report(test_generator_flow.classes, pred))

    print("---------------")
