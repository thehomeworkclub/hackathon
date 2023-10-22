import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model, load_model
from keras.layers import Dense, Flatten
from keras.optimizers import Adam

# Parameters
input_shape = (224, 224, 3)
batch_size = 16
epochs = 10

# Load VGG16 model without top layers
base_model = VGG16(weights='imagenet', include_top=False,
                   input_shape=input_shape)

# Add custom layers
x = base_model.output
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = Dense(1, activation='sigmoid')(x)  # Binary classification

# Create the full model
model = Model(inputs=base_model.input, outputs=x)

# Uncomment the next line if you want to load a previously trained model
# model_path = "classification_model.h5"
# model = load_model(model_path)

# Freeze the layers of the base model
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer=Adam(lr=0.0001),
              loss='binary_crossentropy', metrics=['accuracy'])

# Data augmentation
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2)  # Use 20% of data for validation

train_generator = train_datagen.flow_from_directory(
    './wales',
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='binary',
    subset='training')  # Training data

validation_generator = train_datagen.flow_from_directory(
    './wales',
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='binary',
    subset='validation')  # Validation data

# Train the model
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size)

# After training, offer to save the model
model_path = "classification_model.h5"
if input("Save model? (yes/no): ").strip().lower() == 'yes':
    model.save(model_path)
    print(f"Model saved to {model_path}")

# Prediction function


def predict_whale(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = np.expand_dims(img_to_array(img), axis=0)
    processed_img = preprocess_input(img_array)
    prediction = model.predict(processed_img)
    # Return True if it's a whale, False otherwise
    return prediction[0][0] > 0.5


# Test the prediction function
if __name__ == "__main__":
    while True:
        try:
            img_path = input(
                "Please enter the path to the image (or type 'exit' to stop): ")
            if img_path.lower() == 'exit':
                break
            result = predict_whale(img_path)
            print("Whale Detected!" if result else "No Whale Detected.")
        except Exception as e:
            print(f"Error: {e}")
