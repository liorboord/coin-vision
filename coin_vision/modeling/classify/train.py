import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf

from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.utils import shuffle
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from coin_vision import config
from coin_vision import plots


def test_gpu():
    import tensorflow as tf
    import time
    with tf.device('/GPU:0'):
        a = tf.random.normal([1000, 1000])
        b = tf.random.normal([1000, 1000])
    start = time.time()
    c = tf.matmul(a, b)
    print("Time taken on GPU:", time.time() - start)
    print("TensorFlow version:", tf.__version__)
    physical_devices = tf.config.list_physical_devices()
    print("Physical Devices:")
    for device in physical_devices:
       print(device)

   # Check if Metal plugin (Apple GPU) is being used
    from tensorflow.python.framework import test_util
    if test_util.IsMlcEnabled():
        print("Metal backend (Apple GPU) is enabled.")
    else:
        print("Metal backend is not enabled.")

def prepare_dataset(path):
  dataset = []
  for label in os.listdir(path):
    label_path = os.path.join(path, label)
    if os.path.isdir(label_path):
      for image_file in os.listdir(label_path):
        image_path = os.path.join(label_path, image_file)
        try:
          img = cv2.imread(image_path)
          if img is not None:
            dataset.append((config.LABELS_STRING_TO_INT[label], img))
        except Exception as e:
          print(f"Error processing image {image_path}: {e}")
  return dataset



# Load and shuffle dataset
def load_data(dataset):
    # Separate images and labels
    labels = np.array([item[0] for item in dataset])
    images = np.array([item[1] for item in dataset])

    # Shuffle the dataset
    images, labels = shuffle(images, labels, random_state=42)

    # Normalize the images
    images = images / 255.0

    # Convert labels to categorical format
    num_classes = len(np.unique(labels))
    labels = to_categorical(labels, num_classes)

    return images, labels, num_classes


def create_transfer_model(input_shape=(config.SINGLE_COIN_IMAGE_SIZE, config.SINGLE_COIN_IMAGE_SIZE, 3), num_classes=config.NUMBER_OF_CLASSES):
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,  
        weights='imagenet'  
    )
    base_model.trainable = False  # Freeze the base model layers

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),  # Dropout layer with a rate of 0.5
        layers.Dense(config.NUMBER_OF_CLASSES, activation='softmax')
    ])

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def display_and_save_misclassified_images(model, X_test, y_test, labels_dict, save_folder="../data/interim/misclassified_images"):
    os.makedirs(save_folder, exist_ok=True)

    # Get model predictions
    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)

    # Identify misclassified indices
    misclassified_indices = np.where(y_pred != y_true)[0]
    print(f"Total misclassified images: {len(misclassified_indices)}")

    # Display and save each misclassified image
    for idx in misclassified_indices:
        img = X_test[idx]
        true_label = labels_dict[y_true[idx]]
        predicted_label = labels_dict[y_pred[idx]]

        # Display the image with true and predicted labels
        plt.imshow(img.squeeze(), cmap="gray" if img.shape[-1] == 1 else None)
        plt.title(f"True: {true_label} | Predicted: {predicted_label}")
        plt.axis('off')
        plt.show()

        # Save the misclassified image with labels in the filename
        misclassified_filename = os.path.join(save_folder,
                                              f"misclassified_{idx}_true_{true_label}_pred_{predicted_label}.png")
        plt.imsave(misclassified_filename, img.squeeze(), cmap="gray" if img.shape[-1] == 1 else None)
        print(f"Saved misclassified image: {misclassified_filename}")


def create_and_train_model(X_train, y_train, num_classes):
    input_shape = (config.SINGLE_COIN_IMAGE_SIZE, config.SINGLE_COIN_IMAGE_SIZE, 3)
    model = create_transfer_model(input_shape=input_shape, num_classes=num_classes)

    class_labels = np.argmax(y_train, axis=1)

    # Calculate class weights
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(class_labels), y=class_labels)
    class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}

    print("Class Weights:", class_weights_dict)

    # Further split the train + validation into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    history = model.fit(X_train, y_train, epochs=30, batch_size=16, validation_split=0.2,
                         class_weight=class_weights_dict)

    return model, history

def create_training_data():
    dataset = prepare_dataset(config.LABELED_SINGLE_COINS_FOLDER)
    print(f"Dataset size: {len(dataset)}")

    # Load and preprocess data
    images, labels, num_classes = load_data(dataset)

    # Reshape images if they are grayscale (64, 64) to (64, 64, 1)
    if len(images.shape) == 3:
        images = np.expand_dims(images, -1)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.1, random_state=42)
    return X_train, X_test, y_train, y_test

def create_and_test_model():
    X_train, X_test, y_train, y_test = create_training_data()
    # Create and train the model
    model, history = create_and_train_model(X_train, y_train, config.NUMBER_OF_CLASSES)

    test_loss, test_accuracy = model.evaluate(X_test, y_test)

    print(f"Test Accuracy: {test_accuracy:.2f}")
    display_and_save_misclassified_images(model, X_test, y_test, config.LABELS_INT_TO_STRING)
    plots.plot_and_save_history(history, test_loss, test_accuracy, config.LABELS_INT_TO_STRING)
    model.save(os.path.join(config.MODELS_FOLDER, "coin_classification_model.h5"))
