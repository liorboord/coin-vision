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

import os
import matplotlib.pyplot as plt
import pandas as pd

def plot_and_save_run_results(results, save_folder="reports"):
    """
    Generate and save various plots to evaluate model performance.
    
    Args:
        results (DataFrame): A DataFrame with columns:
            - 'filename': The name of the image file.
            - 'coin_value': Predicted total coin value.
            - 'true_coin_value': Actual total coin value.
            - 'detected_classes': List of detected classes for each image.
        save_folder (str): Folder where the plots will be saved.
    """
    os.makedirs(save_folder, exist_ok=True)
    
    plt.style.use('dark_background')
    primary_color = '#58a6ff'  # Blue
    secondary_color = '#8b949e'  # Grey
    accent_color = '#c9d1d9'  # White
    
    # Absolute Error
    results['absolute_error'] = abs(results['coin_value'] - results['true_coin_value'])
    
    # Percentage Error (handling division by zero)
    results['percentage_error'] = (results['absolute_error'] / results['true_coin_value'].replace(0, 1)) * 100

    # Plot 1: Scatter Plot - Predicted vs True Coin Values
    plt.figure(figsize=(8, 6))
    plt.scatter(results['true_coin_value'], results['coin_value'], alpha=0.7, label="Predictions")
    plt.plot(
        [results['true_coin_value'].min(), results['true_coin_value'].max()],
        [results['true_coin_value'].min(), results['true_coin_value'].max()],
        color='red', linestyle='--', label="Perfect Prediction"
    )
    plt.title("Predicted vs True Coin Values")
    plt.xlabel("True Coin Value")
    plt.ylabel("Predicted Coin Value")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(save_folder, "scatter_predicted_vs_true.png"))
    plt.close()

    # Plot 2: Bar Plot - Absolute Error per Image
    plt.figure(figsize=(10, 6))
    plt.bar(results['filename'], results['absolute_error'], color='orange', alpha=0.8)
    plt.title("Absolute Error per Image")
    plt.xlabel("Image Filename")
    plt.ylabel("Absolute Error")
    plt.xticks(rotation=90, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, "absolute_error_per_image.png"))
    plt.close()

    # Plot 3: Histogram - Percentage Error Distribution
    plt.figure(figsize=(8, 6))
    plt.hist(results['percentage_error'], bins=20, color='blue', alpha=0.7)
    plt.title("Percentage Error Distribution")
    plt.xlabel("Percentage Error")
    plt.ylabel("Frequency")
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(save_folder, "percentage_error_distribution.png"))
    plt.close()

    # Plot 4: Cumulative Sum - True vs Predicted Coin Values
    results['cumulative_true'] = results['true_coin_value'].cumsum()
    results['cumulative_predicted'] = results['coin_value'].cumsum()
    plt.figure(figsize=(10, 6))
    plt.plot(results['cumulative_true'], label="Cumulative True Values", color='green')
    plt.plot(results['cumulative_predicted'], label="Cumulative Predicted Values", linestyle='--', color='blue')
    plt.title("Cumulative True vs Predicted Coin Values")
    plt.xlabel("Image Index")
    plt.ylabel("Cumulative Coin Value")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(save_folder, "cumulative_true_vs_predicted.png"))
    plt.close()

    # Save Metrics Summary as CSV
    metrics_summary = {
        "Mean Absolute Error": results['absolute_error'].mean(),
        "Overall Accuracy": 1 - (abs(results['coin_value'].sum() - results['true_coin_value'].sum()) / results['true_coin_value'].sum()),
        "R-Squared": (1 - ((results['coin_value'] - results['true_coin_value'])**2).sum() /
                        ((results['true_coin_value'] - results['true_coin_value'].mean())**2).sum())
    }
    metrics_df = pd.DataFrame.from_dict(metrics_summary, orient='index', columns=["Value"])
    metrics_df.to_csv(os.path.join(save_folder, "metrics_summary.csv"))

    print(f"Plots and metrics saved in {save_folder}")

