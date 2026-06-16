import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
train_ds = tf.keras.utils.image_dataset_from_directory(
    r"C:\Users\panav\Downloads\MRL\data\train",
    image_size=(86, 86),
    color_mode="grayscale",
    batch_size=32
)
print(train_ds.class_names)
model=keras.Sequential([
    layers.Input(shape=(86,86,1)), #image shape =86,86 ; CNN expects channel: grayscale=1 channel
    layers.Rescaling(1./255),
    layers.Conv2D(
        32,
        (3,3),
        activation='relu'
    ),
    layers.MaxPooling2D(),
    layers.Conv2D(
        64,
        (3,3),
        activation='relu'
    ),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(
        128,
        activation='relu'
    ),
    layers.Dense(
        1,
        activation='sigmoid'
    )

])
model.summary()