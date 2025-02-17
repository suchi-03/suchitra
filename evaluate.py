import tensorflow as tf
from tensorflow.keras.models import load_model
from data_preprocessing import validation_generator

model = load_model('potato_leaf_disease_model.h5')

loss, accuracy = model.evaluate(validation_generator)
print(f'Validation Loss: {loss}')
print(f'Validation Accuracy: {accuracy}')