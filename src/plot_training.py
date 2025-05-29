import matplotlib.pyplot as plt

def plot_training(history, model_name="model"):
    import matplotlib.pyplot as plt

    if isinstance(history, list):
        acc = history[0].history['accuracy']
        val_acc = history[0].history['val_accuracy']
        loss = history[0].history['loss']
        val_loss = history[0].history['val_loss']

        for h in history[1:]:
            acc += h.history['accuracy']
            val_acc += h.history['val_accuracy']
            loss += h.history['loss']
            val_loss += h.history['val_loss']
    else:
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, label='Train Accuracy')
    plt.plot(epochs, val_acc, label='Validation Accuracy')
    plt.title(f'{model_name} Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, label='Train Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.title(f'{model_name} Loss')
    plt.legend()

    plt.savefig(f"{model_name}_training.png")
    plt.show()
