import os
import numpy as np
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Lambda, Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras import backend as K
from keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator

# Define the base network (sub-network) for the Siamese Network


def create_base_network(input_shape):
    input = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu')(input)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    return Model(input, x)


# Create the Siamese Network
input_shape = (224, 224, 3)
base_network = create_base_network(input_shape)

input_a = Input(shape=input_shape)
input_b = Input(shape=input_shape)

processed_a = base_network(input_a)
processed_b = base_network(input_b)

# Compute the absolute difference between the feature vectors
distance = Lambda(lambda tensors: K.abs(
    tensors[0] - tensors[1]))([processed_a, processed_b])

# Produce the final similarity score
prediction = Dense(1, activation='sigmoid')(distance)

siamese_net = Model([input_a, input_b], prediction)

siamese_net.compile(loss='binary_crossentropy',
                    optimizer=Adam(lr=0.0001), metrics=['accuracy'])


def prepare_data_pairs(whale_folder):
    whale_images = []
    for img in os.listdir(whale_folder):
        img_path = os.path.join(whale_folder, img)
        try:
            whale_images.append(img_to_array(
                load_img(img_path, target_size=input_shape)))
        except Exception as e:
            print(f"Error loading {img}: {e}")
            continue  # Skip this image and continue with the next one
    return np.array(whale_images)


whale_data = prepare_data_pairs('./wales/whale')

# This function will prepare the input pairs for the Siamese Network using the provided image path


def create_pairs(image_path):
    image = img_to_array(load_img(image_path, target_size=input_shape))
    pairs = [[image, whale_img] for whale_img in whale_data]
    return np.array(pairs)

# Predicting if an image contains a whale


def contains_whale(image_path, model):
    pairs = create_pairs(image_path)
    predictions = model.predict([pairs[:, 0], pairs[:, 1]])
    # Return True if any of the prediction scores is above a threshold (e.g., 0.5)
    return np.any(predictions > 0.5)


if __name__ == "__main__":
    while True:
        try:
            img_path = input(
                "Please enter the path to the image (or type 'exit' to stop): ")
            if img_path.lower() == 'exit':
                break
            result = contains_whale(img_path, siamese_net)
            print(result)
        except Exception as e:
            print(f"Error: {e}")
