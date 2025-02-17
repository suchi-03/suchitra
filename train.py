import tensorflow as tf
from model import model
from data_preprocessing import train_generator, validation_generator

epochs = 25

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=epochs
)

model.save('potato_leaf_disease_model.h5')