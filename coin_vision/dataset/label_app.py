import os
import cv2
import shutil
import coin_vision.config

# Define the labels and destination folder
#LABELS = ["1T", "1H", "2T", "2H", "5T", "5H", "10T", "10H", "None"]
#DESTINATION_FOLDER = "../../data/processed/labeled_single_coin_2"
#SOURCE_FOLDER = '../../data/interim/detection_results'

LABELS = {
    ord('1'): "1T",
    ord('2'): "1H",
    ord('3'): "2T",
    ord('4'): "2H",
    ord('5'): "5T",
    ord('6'): "5H",
    ord('7'): "10T",
    ord('8'): "10H",
    ord('9'): "None"
}


# Function to label images
def label_images(source_folder, destination_folder):
    # Ensure destination folders are set up
    _create_label_folders(destination_folder, LABELS)

    # Go through each image in the input folder
    for filename in os.listdir(source_folder):
        # Check if file is an image
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            file_path = os.path.join(source_folder, filename)

            # Load and display the image
            image = cv2.imread(file_path)
            if image is None:
                print(f"Could not open {filename}. Skipping...")
                continue

            # Resize image if too large
            max_dimension = 800  # Resize to max 800px if necessary
            height, width = image.shape[:2]
            if max(height, width) > max_dimension:
                scaling_factor = max_dimension / max(height, width)
                image = cv2.resize(image, (int(width * scaling_factor), int(height * scaling_factor)))

            # Display the image in a resizable window
            cv2.namedWindow("Image", cv2.WINDOW_NORMAL)  # Allow window resizing
            cv2.imshow("Image", image)
            print(f"Labeling: {filename}")
            print("Press a key to label the image:")
            for key, label in LABELS.items():
                print(f"  Press {chr(key)} for {label}")

            # Wait for user to press a key and get the label
            key = cv2.waitKey(0)  # Wait indefinitely for a key press
            if key in LABELS:
                label = LABELS[key]
                destination_path = os.path.join(destination_folder, label, filename)
                shutil.move(file_path, destination_path)
                print(f"Moved {filename} to {label} folder.")
            else:
                print(f"Invalid key pressed for {filename}. Skipping...")

            # Close the OpenCV image window
            cv2.destroyAllWindows()

    print("Labeling completed for all images.")

## Function to ensure label folders exist
def _create_label_folders(destination_folder, labels):
    os.makedirs(destination_folder, exist_ok=True)
    for label in labels.values():
        os.makedirs(os.path.join(destination_folder, label), exist_ok=True)

        
# Example usage
#input_folder = "./coin_images"  # Replace with the path to your folder of images
#label_images(SOURCE_FOLDER)
