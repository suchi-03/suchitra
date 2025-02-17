import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

model = load_model('potato_leaf_disease_model.h5')

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction, axis=1)[0]
    class_labels = list(validation_generator.class_indices.keys())
    return class_labels[class_idx]

# Example usage
img_path = 'path/to/your/test_image.jpg'
predicted_class = predict_image(img_path)
print(f'Predicted class: {predicted_class}')