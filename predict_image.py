import tensorflow as tf
import numpy as np
from tensorflow import keras

#loading trained model:
model=keras.models.load_model("drowsiness_model.keras")
print("Model loaded successfully!")

#loading image:
image_path = r"C:\Users\panav\Downloads\MRL\data\test\sleepy\s0037_05865_1_1_0_0_1_01.png"
img=tf.keras.utils.load_img(
    image_path,
    color_mode="grayscale",
    target_size=(86,86)
)
img_array=tf.keras.utils.img_to_array(img)
print("Image Shape:", img_array.shape)
img_array = np.expand_dims(img_array, axis=0)
print("Batch Shape:", img_array.shape)
prediction = model.predict(img_array)

score=prediction[0][0]
if score<0.5:   #decision threshold
    confidence=(1-score)*100
    print(f"Prediction: Awake")
    print(f"confidence: {confidence:.2f}%")
else:
    confidence = score * 100
    print(f"Prediction: SLEEPY")
    print(f"Confidence: {confidence:.2f}%")
