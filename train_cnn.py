import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# Load Training Dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
    r"C:\Users\panav\Downloads\MRL\data\train",
    image_size=(86, 86),
    color_mode="grayscale",
    batch_size=32,
    label_mode="binary"
)
# Validation Dataset
val_ds = tf.keras.utils.image_dataset_from_directory(
    r"C:\Users\panav\Downloads\MRL\data\val",
    image_size=(86, 86),
    color_mode="grayscale",
    batch_size=32,
    label_mode="binary"
)
print("classes: ",train_ds.class_names)


AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

#cnn model:
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

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
#train model
history=model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=2
)


model.save("drowsiness_model.keras")
