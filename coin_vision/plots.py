import os
import matplotlib.pyplot as plt
from coin_vision import config

def plot_and_save_history(history, test_loss, test_accuracy, labels_dict, save_folder=config
                          .REPORTS_FOLDER):
    os.makedirs(save_folder, exist_ok=True)

    # Plot training & validation accuracy
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_folder, 'accuracy_plot.png'))
    plt.show()

    # Plot training & validation loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_folder, 'loss_plot.png'))
    plt.show()

    # Write test results to a file
    results_path = os.path.join(save_folder, "test_results.txt")
    with open(results_path, "w") as f:
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Test Accuracy: {test_accuracy:.4f}\n")

    # Generate confusion matrix
    # y_pred = np.argmax(model.predict(X_test), axis=1)
    # y_true = np.argmax(y_test, axis=1)
    # cm = confusion_matrix(y_true, y_pred)
    #
    # # Plot and save confusion matrix
    # plt.figure(figsize=(8, 8))
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm,
    #                               display_labels=[labels_dict[i] for i in range(len(labels_dict))])
    # disp.plot(cmap="Blues", values_format="d", ax=plt.gca())
    # plt.title("Confusion Matrix")
    # cm_path = os.path.join(save_folder, "confusion_matrix.png")
    # plt.savefig(cm_path)
    # plt.show()

    print(f"Plots and results saved in folder: {save_folder}")