import cv2
import numpy as np
import os
import config

INPUT_FOLDER = "../static/coin_images_raw"  # Folder containing images with multiple coins
OUTPUT_FOLDER = "../static/coin_images_extracted"  # Folder to save extracted coin images

def detect_and_extract_coins_from_folder(input_folder, output_folder, output_size=config.SINGLE_COIN_IMAGE_SIZE):
    # Create output directories if they don't exist
    os.makedirs(output_folder, exist_ok=True)

    # Iterate over all files in the input folder
    for filename in os.listdir(input_folder):
        # Build full file path
        file_path = os.path.join(input_folder, filename)

        # Check if file is an image (you can add more extensions if needed)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            print(f"Processing {filename}")

            # Read the image in color
            color_image = cv2.imread(file_path)
            gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)  # Grayscale for circle detection

            # Apply Gaussian Blur to reduce noise for circle detection
            blurred = cv2.GaussianBlur(gray_image, (11, 11), 0)

            # Detect circles using Hough Circle Transform
            circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
                                       param1=100, param2=30, minRadius=20, maxRadius=100)

            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")

                for i, (x, y, r) in enumerate(circles):
                    # Crop the coin from the original color image to keep colors
                    coin_color = color_image[y - r:y + r, x - r:x + r]
                    if coin_color.shape[0] > 0 and coin_color.shape[1] > 0:
                        coin_resized_color = cv2.resize(coin_color, output_size)

                        # Save the color version
                        color_filename = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_coin_{i}.jpg")
                        cv2.imwrite(color_filename, coin_resized_color)
                        print(f"Saved color coin: {color_filename}")

#detect_and_extract_coins_from_folder(INPUT_FOLDER, OUTPUT_FOLDER)



