import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load trained model
model = tf.keras.models.load_model("model/image_tagger.h5")

# Class labels (same order as dataset folders)
class_names = ['cat', 'dog']

# Load and preprocess image
img = image.load_img("dataset/cat/cat1.jpg", target_size=(128, 128))
img = image.img_to_array(img) / 255.0
img = np.expand_dims(img, axis=0)

# Predict
prediction = model.predict(img)
predicted_index = np.argmax(prediction)
predicted_class = class_names[predicted_index]

print("Prediction probabilities:", prediction)
print("Predicted class:", predicted_class)
