import os
import numpy as np
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.callbacks import EarlyStopping

# Load VGG16 without the top classification layers
base_model = VGG16(weights='imagenet', include_top=False)

# Add custom layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)  # Add dropout
predictions = Dense(2, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze VGG16 layers
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=Adam(lr=0.0001),
              loss='categorical_crossentropy', metrics=['accuracy'])

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    validation_split=0.2,
    preprocessing_function=preprocess_input
)

train_generator = datagen.flow_from_directory(
    './wales',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    './wales',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Early stopping callback
early_stop = EarlyStopping(
    monitor='val_loss', patience=5, restore_best_weights=True)

model.fit_generator(
    train_generator,
    epochs=10,
    validation_data=validation_generator,
    callbacks=[early_stop]  # Add early stopping callback
)


def contains_whale(input_img_path, model):
    img = image.load_img(input_img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    prediction = model.predict(preprocessed_img)
    # Assuming whale class is 1 and no-whale is 0
    return prediction[0][1] > 0.5


if __name__ == "__main__":
    while True:
        try:
            img_path = input(
                "Please enter the path to the image (or type 'exit' to stop): ")
            if img_path.lower() == 'exit':
                break
            result = contains_whale(img_path, model)
            print(result)
        except Exception as e:
            print(f"Error: {e}")
