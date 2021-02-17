import numpy as np
import time

import PIL.Image as Image
import matplotlib.pylab as plt

import tensorflow as tf
import tensorflow_hub as hub

def main():
    # Resnet 50
    feature_extractor_model = "https://tfhub.dev/tensorflow/resnet_50/feature_vector/1"  # @param {type:"string"}
    feature_extractor_layer = hub.KerasLayer(
        # trainable = False freezes the variables in feature extractor layer,
        # so that the training only modifies the new classifier layer.
        feature_extractor_model, input_shape=(224, 224, 3), trainable=False)

    image_batch = np.load('Data/training.npy', allow_pickle=True)

    # It returns a 1280-length vector for each image:
    feature_batch = feature_extractor_layer(image_batch)
    print(feature_batch.shape)

    # Attach a classification head
    num_classes = len(class_names)

    # Now wrap the hub layer in a tf.keras.Sequential model
    model = tf.keras.Sequential([
        feature_extractor_layer,
        tf.keras.layers.Dense(num_classes)  # add a new classification layer.
    ])

    model.summary()

if __name__ == "__main__":
    main()