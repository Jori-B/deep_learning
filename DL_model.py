import numpy as np
import time

import PIL.Image as Image
import matplotlib.pylab as plt

import tensorflow as tf
import tensorflow_hub as hub

IMAGE_SHAPE = (224, 224)

def main():
    batch_size = 32  # the number of training examples utilized in one iteration
    img_height = 224
    img_width = 224
    num_epoch = 100

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        str('PokemonData'),
        validation_split=0.2,  # Fraction of the training data to be used as validation data.
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    class_names = np.array(train_ds.class_names)
    print(class_names)
    print(train_ds)

    normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)

    for image_batch, labels_batch in train_ds:
        print(image_batch.shape)
        print(labels_batch.shape)
        break
    # Attach a classification head
    num_classes = len(class_names)

    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(filters=64, kernel_size=4, activation='relu', input_shape=(224, 224, 3)),
        tf.keras.layers.Conv1D(filters=32, kernel_size=4, activation='relu'),
        tf.keras.layers.Conv1D(filters=16, kernel_size=4, activation='relu'),
        tf.keras.layers.AveragePooling2D(pool_size = 2, strides = 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes)  # add a new classification layer.
    ])
    #model.summary()

    predictions = model(image_batch)
    predictions.shape

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['acc'])

    class CollectBatchStats(tf.keras.callbacks.Callback):
        def __init__(self):
            self.batch_losses = []
            self.batch_acc = []

        def on_train_batch_end(self, batch, logs=None):
            self.batch_losses.append(logs['loss'])
            self.batch_acc.append(logs['acc'])
            self.model.reset_metrics()

    batch_stats_callback = CollectBatchStats()

    history = model.fit(train_ds, epochs=num_epoch,
                        callbacks=[batch_stats_callback])

    plt.figure()
    plt.ylabel("Loss")
    plt.xlabel("Training Steps")
    plt.ylim([0, 2])
    plt.plot(batch_stats_callback.batch_losses)
    plt.savefig('Loss.png')


    plt.figure()
    plt.ylabel("Accuracy")
    plt.xlabel("Training Steps")
    plt.ylim([0, 1])
    plt.plot(batch_stats_callback.batch_acc)
    plt.savefig('Accuracy.png')

    predicted_batch = model.predict(image_batch)
    predicted_id = np.argmax(predicted_batch, axis=-1)
    predicted_label_batch = class_names[predicted_id]

    plt.figure(figsize=(10, 9))
    plt.subplots_adjust(hspace=0.5)
    for n in range(30):
        plt.subplot(6, 5, n + 1)
        plt.imshow(image_batch[n])
        plt.title(predicted_label_batch[n].title())
        plt.axis('off')
    _ = plt.suptitle("Model predictions")
    plt.savefig('predicted_images.png')


if __name__ == "__main__":
    main()
