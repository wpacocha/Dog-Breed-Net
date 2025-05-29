import matplotlib.pyplot as plt

def plot_training(history, model_name="model"):
    # Jeśli podano tylko 1 obiekt
    if not isinstance(history, list):
        history = [history]

    # Wyodrębnij metryki
    acc = []
    val_acc = []
    loss = []
    val_loss = []

    for h in history:
        if hasattr(h, "history"):
            acc += h.history.get('accuracy', [])
            val_acc += h.history.get('val_accuracy', [])
            loss += h.history.get('loss', [])
            val_loss += h.history.get('val_loss', [])

    epochs = range(1, len(acc) + 1)

    # Wykres
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
