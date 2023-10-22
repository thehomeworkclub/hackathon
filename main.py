import os
import numpy as np
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Lambda, Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras import backend as K
from keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator
from keras.layers import Dropout
from keras.models import load_model


def create_base_network(input_shape):
    input = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu')(input)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    return Model(input, x)


def contrastive_loss(y_true, y_pred):
    margin = 1
    y_true = K.cast(y_true, 'float32')  # Ensure y_true is float32
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


input_shape = (224, 224, 3)
base_network = create_base_network(input_shape)

input_a = Input(shape=input_shape)
input_b = Input(shape=input_shape)

processed_a = base_network(input_a)
processed_b = base_network(input_b)

distance = Lambda(lambda tensors: K.abs(
    tensors[0] - tensors[1]))([processed_a, processed_b])
prediction = Dense(1, activation='sigmoid')(distance)

siamese_net = Model([input_a, input_b], prediction)
siamese_net.compile(loss=contrastive_loss, optimizer=Adam(lr=0.00005))


def prepare_data_pairs(folder):
    images = []
    for img in os.listdir(folder):
        img_path = os.path.join(folder, img)
        try:
            images.append(img_to_array(
                load_img(img_path, target_size=input_shape)))
        except Exception as e:
            print(f"Error loading {img}: {e}")
            continue
    return np.array(images)


def create_pairs(whale_data, non_whale_data, num_pairs=1000):
    pairs = []
    labels = []

    for _ in range(num_pairs // 2):
        idx1, idx2 = np.random.choice(len(whale_data), 2, replace=False)
        pairs.append([whale_data[idx1], whale_data[idx2]])
        labels.append(1)

    for _ in range(num_pairs // 2):
        idx1 = np.random.choice(len(whale_data))
        idx2 = np.random.choice(len(non_whale_data))
        pairs.append([whale_data[idx1], non_whale_data[idx2]])
        labels.append(0)

    return np.array(pairs), np.array(labels)


whale_data = prepare_data_pairs('./wales/whale')
# Assuming you have this directory
non_whale_data = prepare_data_pairs('./wales/non_whale')

pairs, labels = create_pairs(whale_data, non_whale_data)

# Train the model
siamese_net.fit([pairs[:, 0], pairs[:, 1]], labels, epochs=10, batch_size=16)

# Save the model
model_path = "siamese_model.h5"
if input("Save model? (yes/no): ").strip().lower() == 'yes':
    siamese_net.save(model_path)
    print(f"Model saved to {model_path}")


def contains_whale(image_path, model):
    image = img_to_array(load_img(image_path, target_size=input_shape))
    pairs = [[image, whale_img] for whale_img in whale_data]
    predictions = model.predict([pairs[:, 0], pairs[:, 1]])
    return np.any(predictions < 0.5)


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
