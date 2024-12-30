import argparse
from . import config
from .modeling.detect import coin_ditector
from .modeling.classify.train import create_and_test_model 
from .modeling.classify.train import test_gpu
from .dataset.label_app import label_images
from .run_detect_and_classify import calculate_image_coin_value

def main():
    # # Parse command-line arguments
    parser = argparse.ArgumentParser(description="CoinVision Application")
    parser.add_argument(
        '--action',
        required=True,
        help='Action to perform: detect, train, evaluate...'
    )
    args = parser.parse_args()
    action = args.action
    #action = 'detect'
    if action == 'detect':
        print('Detecting coins...')
        coin_ditector.handle_detected_coins(config.RAW_IMAGES_FOLDER, config.DETECTION_RESULTS_FOLDER, (config.SINGLE_COIN_IMAGE_SIZE, config.SINGLE_COIN_IMAGE_SIZE), save_mode='0')

    if action == 'label':
        print('Labeling images...')
        label_images(config.DETECTION_RESULTS_FOLDER, config.LABELED_SINGLE_COINS_FOLDER)

    if action == 'train':
        print('Training and testing...')
        create_and_test_model()

    if action == 'test_gpu':
        print('Testing GPU...')
        test_gpu()    

    if action == 'run':
        print('Labeling images...')
        calculate_image_coin_value(config.LABELED_MULTI_COIN_FOLDER)

if __name__ == "__main__":
    main()
