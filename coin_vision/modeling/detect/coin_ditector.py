from ultralytics import YOLO
import cv2
from pathlib import Path
import os
import shutil
import sys
from . import filters
from coin_vision import config

# Load YOLOv8 pretrained model (replace with a custom model if available)
model = YOLO('../models/yolov8s.pt')  # You can use 'yolov8n.pt' for a faster, smaller model


#INPUT_FOLDER = "static/raw_samples"  # Folder containing images with multiple coins
#OUTPUT_FOLDER = "static/yolo_detected"  # Folder to save extracted coin images

DETECTION_CONFIDENCE_THRESHOLD = 0.001
IOU_THRESHOLD = 0.05


def detect_and_extract_coins(input_folder, output_size=(config.SINGLE_COIN_IMAGE_SIZE, config.SINGLE_COIN_IMAGE_SIZE)):
    extracted_images = {}
    for filename in os.listdir(input_folder):
        file_images = detect_and_extract_coins_from_file(input_folder, filename, output_size)
        extracted_images[filename] = file_images
    return extracted_images


def detect_and_extract_coins_from_file(input_folder, filename, output_size, iou_threshold=0.05):
    extracted_images = []
    file_path = os.path.join(input_folder, filename)

    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        print(f"Processing {filename}")

        # Read the image
        color_image = cv2.imread(file_path)
        if color_image is None:
            print(f"Could not open {filename}. Skipping...")
            return []

        # Detect coins using YOLOv8
        try:
            results = model(file_path, conf=DETECTION_CONFIDENCE_THRESHOLD)
        except:
            print("Error during YOLOv8 detection.")
            return []

        # Extract bounding boxes for detected coins, with IoU filtering
        filtered_boxes = filters.filter_duplicates(results, iou_threshold)
        filtered_boxes = filters.filter_round_or_oval_detections(filtered_boxes)

        # Process and save non-duplicate detections
        for i, (xmin, ymin, xmax, ymax) in enumerate(filtered_boxes):
            # Crop the detected coin region
            coin_color = color_image[ymin:ymax, xmin:xmax]
            # Resize to the desired output size
            coin_resized_color = cv2.resize(coin_color, output_size)
            extracted_images.append((coin_resized_color, filename, i))

    return extracted_images


def handle_detected_coins(input_folder, output_folder, output_size, mode='save_results', save_mode='by_file'):
    """
    Handles detected coins by either saving the results to the output folder or returning them as a list.

    Args:
        input_folder (str): Path to the input folder containing images.
        output_folder (str): Path to the output folder for saving results.
        output_size (tuple): Desired output size for resized images.
        mode (str): Mode of operation ('save_results' or 'return_results').
        save_mode (str): Save mode ('by_file' for subfolders or 'flat' for all in the output folder).
    """
    if mode == 'save_results':
        os.makedirs(output_folder, exist_ok=True)

    # Extract coins from images
    extracted_images = detect_and_extract_coins(input_folder, output_size)

    if mode == 'save_results':
        for filename, images in extracted_images.items():
            if images:
                parent_folder = None
                if save_mode == 'by_file':
                    # Create a subfolder named after the parent file
                    parent_folder = os.path.join(output_folder, Path(filename).stem)
                    os.makedirs(parent_folder, exist_ok=True)

                for i, coin_resized_color in enumerate(images):
                    # Determine the save path
                    if save_mode == 'by_file' and parent_folder:
                        color_filename = os.path.join(parent_folder, f"coin_{i}.jpg")
                    else:
                        color_filename = os.path.join(output_folder, f"{Path(filename).stem}_coin_{i}.jpg")

                    # Save the image
                    cv2.imwrite(color_filename, coin_resized_color[0])

                if save_mode == 'by_file' and parent_folder:
                    # Save the original image in the subfolder
                    original_image_basename = Path(filename).name
                    src_path = os.path.join(input_folder, filename)
                    dest_path = os.path.join(parent_folder, 'orig.jpg')
                    shutil.copy(str(src_path), dest_path)
    elif mode == 'return_results':
        # Return a flat list of images
        return [coin_resized_color for images in extracted_images.values() for coin_resized_color in images]



#handle_detected_coins(config.RAW_IMAGES_FOLDER, config.DETECTION_RESULTS_FOLDER, (config.SINGLE_COIN_IMAGE_SIZE, config.SINGLE_COIN_IMAGE_SIZE))

#detect_and_extract_coins(INPUT_FOLDER)
