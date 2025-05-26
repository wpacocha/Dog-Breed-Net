import matplotlib.pyplot as plt

def plot_training(history,fine_tune_history=None):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss =history.history['val_loss']

    if fine_tune_history:
        acc += fine_tune_history.history['accuracy']
        val_acc += fine_tune_history.history['val_accuracy']
        loss += fine_tune_history.history['loss']
        val_loss += fine_tune_history.history['val_loss']

    epochs = range(1, len(acc)+1)

    plt.figure(figsize = (12,5))
    plt.subplot(1,2,1)
    plt.plot(epochs,loss, 'bo-', label='Train loss')
    plt.plot(epochs,val_loss,'r^-', label='Val loss')
    plt.legend()
    plt.title('Loss')
    plt.show()