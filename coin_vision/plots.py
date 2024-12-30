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


    print(f"Plots and results saved in folder: {save_folder}")

def plot_and_save_run_results(results, save_folder="reports"):
    """
    Generate and save various plots to evaluate model performance with a custom dark theme.
    
    Args:
        results (DataFrame): A DataFrame with columns:
            - 'filename': The name of the image file.
            - 'coin_value': Predicted total coin value.
            - 'true_coin_value': Actual total coin value.
            - 'detected_classes': List of detected classes for each image.
        save_folder (str): Folder where the plots will be saved.
    """
    os.makedirs(save_folder, exist_ok=True)
    
    # Define custom colors
    background_color = '#0D1117'  # Dark background
    primary_color = '#58a6ff'  # Blue
    secondary_color = '#8b949e'  # Grey
    accent_color = '#c9d1d9'  # White

    # Set figure defaults
    plt.rcParams.update({
        'axes.facecolor': background_color,
        'figure.facecolor': background_color,
        'axes.edgecolor': secondary_color,
        'axes.labelcolor': accent_color,
        'xtick.color': accent_color,
        'ytick.color': accent_color,
        'grid.color': secondary_color,
        'text.color': accent_color
    })

    # Absolute Error
    results['absolute_error'] = abs(results['coin_value'] - results['true_coin_value'])
    
    # Percentage Error (handling division by zero)
    results['percentage_error'] = (results['absolute_error'] / results['true_coin_value'].replace(0, 1)) * 100

    # Plot 1: Scatter Plot - Predicted vs True Coin Values
    plt.figure(figsize=(8, 6))
    plt.scatter(results['true_coin_value'], results['coin_value'], alpha=0.7, color=primary_color, label="Predictions")
    plt.plot(
        [results['true_coin_value'].min(), results['true_coin_value'].max()],
        [results['true_coin_value'].min(), results['true_coin_value'].max()],
        color=secondary_color, linestyle='--', label="Perfect Prediction"
    )
    plt.title("Predicted vs True Coin Values")
    plt.xlabel("True Coin Value")
    plt.ylabel("Predicted Coin Value")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(save_folder, "scatter_predicted_vs_true.png"), transparent=True)
    plt.close()

    # Plot 2: Bar Plot - Absolute Error per Image
    plt.figure(figsize=(10, 6))
    plt.bar(results['filename'], results['absolute_error'], color=primary_color, alpha=0.8)
    plt.title("Absolute Error per Image")
    plt.xlabel("Image Filename")
    plt.ylabel("Absolute Error")
    plt.xticks(rotation=90, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, "absolute_error_per_image.png"), transparent=True)
    plt.close()

    # Plot 3: Histogram - Percentage Error Distribution
    plt.figure(figsize=(8, 6))
    plt.hist(results['percentage_error'], bins=20, color=primary_color, alpha=0.7)
    plt.title("Percentage Error Distribution")
    plt.xlabel("Percentage Error")
    plt.ylabel("Frequency")
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(save_folder, "percentage_error_distribution.png"), transparent=True)
    plt.close()

    # Plot 4: Cumulative Sum - True vs Predicted Coin Values
    results['cumulative_true'] = results['true_coin_value'].cumsum()
    results['cumulative_predicted'] = results['coin_value'].cumsum()
    plt.figure(figsize=(10, 6))
    plt.plot(results['cumulative_true'], label="Cumulative True Values", color=primary_color)
    plt.plot(results['cumulative_predicted'], label="Cumulative Predicted Values", linestyle='--', color=secondary_color)
    plt.title("Cumulative True vs Predicted Coin Values")
    plt.xlabel("Image Index")
    plt.ylabel("Cumulative Coin Value")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(save_folder, "cumulative_true_vs_predicted.png"), transparent=True)
    plt.close()

    print(f"Plots saved in {save_folder}")

