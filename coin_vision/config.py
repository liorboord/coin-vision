from tkinter.constants import SINGLE

DATA_FOLDER = 'data'

RAW_IMAGES_FOLDER = f'{DATA_FOLDER}/raw/coin_images_raw'
LABELED_SINGLE_COINS_FOLDER = f'{DATA_FOLDER}/processed/labeled_single_coin'
LABELED_MULTI_COIN_FOLDER = f'{DATA_FOLDER}/raw/labled_full_images'
DETECTION_RESULTS_FOLDER = f'{DATA_FOLDER}/interim/detection_results_single_coin'

MODELS_FOLDER = '../models'
REPORTS_FOLDER = '../reports'

NUMBER_OF_CLASSES = 8
SINGLE_COIN_IMAGE_SIZE = 224

LABELS_STRING_TO_INT = {
  "1H": 0,
  "1T": 1,
  "2H": 2,
  "2T": 3,
  "5T": 4,
  "5H": 5,
  "10T":6,
  "10H": 7
}

# Reverse the dictionary to get int-to-string mapping
LABELS_INT_TO_STRING = {v: k for k, v in LABELS_STRING_TO_INT.items()}
