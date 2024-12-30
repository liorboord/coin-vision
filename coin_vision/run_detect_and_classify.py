import os
import sys
print(sys.path)
import numpy as np
from .modeling.detect.coin_ditector import detect_and_extract_coins_from_file
import cv2
import tensorflow as tf
from coin_vision import config
import pandas as pd

COIN_CLASSIFICATION_PROB_THRESHOLD = 0.9

COIN_LABEL_VALUE = {
  "1H": 1,
  "1T": 1,
  "2H": 2,
  "2T": 2,
  "5T": 5,
  "5H": 5,
  "10T":10,
  "10H": 10
}

def calculate_image_coin_value(folder):
    model = tf.keras.models.load_model(config.MODELS_FOLDER+'/coin_classification_model.h5')
    images_coin_data = []

    for filename in os.listdir(folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):  # Adjust file extensions as needed

            filepath = os.path.join(folder, filename)
            coins = detect_and_extract_coins_from_file(folder, filename, output_size=(config.SINGLE_COIN_IMAGE_SIZE, config.SINGLE_COIN_IMAGE_SIZE))
            coin_data = []
            print('Number of images found:', len(coins))

            # Create a directory for each filename in LABELED_IMAGES_FOLDER
            file_folder = os.path.join(config.DETECTION_RESULTS_FOLDER, os.path.splitext(filename)[0])
            os.makedirs(file_folder, exist_ok=True)

            coin_count = 0
            for coin in coins:
                coin_data.append({
                    'source_image': filename,
                    'coin_image': coin[0],
                    'coin_class': -1,
                    'coin_value': 0
                })

            for coin_info in coin_data:
                image = coin_info['coin_image']
                img_array = image / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                predictions = model.predict(img_array)

                # Get the highest probability and corresponding class index
                max_prob = np.max(predictions)
                predicted_class = np.argmax(predictions)

                # Check if confidence meets the threshold
                if max_prob < COIN_CLASSIFICATION_PROB_THRESHOLD:
                    coin_info['coin_class'] = -1  # Dismiss image if below threshold
                else:
                    coin_info['coin_class'] = predicted_class  # Set the class index if above threshold
                    print('Coin class:', config.LABELS_INT_TO_STRING[coin_info['coin_class']])

                    # Save the image with a new filename containing the class information
                    output_filename = f"object_{coin_count}_class_{config.LABELS_INT_TO_STRING[coin_info['coin_class']]}.png"
                    output_path = os.path.join(file_folder, output_filename)
                    coin_count += 1

                    # Write the coin image to the new file in the specific folder
                    cv2.imwrite(output_path, coin_info['coin_image'])
                    print(f"Saved color coin: {output_path}")    
            images_coin_data.append(get_coin_in_image_value(coin_data, filename))
    # Convert the images_coin_data list of dictionaries into a DataFrame
    df = pd.DataFrame(images_coin_data)
    # Print the DataFrame in a table format
    print(df)
    true_coins_value = 0
    for filename in df['filename']:
        if filename.startswith('f_') and filename.endswith('.jpg'):
            value_str = filename.split('_')[1]
            try:
                value = int(value_str)
                true_coins_value += value
            except ValueError:
                pass
    detected_coins_value = df['coin_value'].sum()
    error = abs(true_coins_value - detected_coins_value)
    print(f"True coins value: {true_coins_value}", f"Detected coins value: {detected_coins_value}", f"Error: {error}")

def get_coin_in_image_value(coin_data, filename):
    image_coins_value = 0
    detected_coins = []
    if coin_data is not None:
        for coin in coin_data:
            if coin['coin_class'] != -1:
                coin_class = config.LABELS_INT_TO_STRING[coin['coin_class']]
                coin_value = COIN_LABEL_VALUE[coin_class]
                image_coins_value += coin_value
                detected_coins.append(coin_class)
    return {
        'filename': filename,
        'coin_value': image_coins_value,
        'detected_classes': detected_coins
    }

