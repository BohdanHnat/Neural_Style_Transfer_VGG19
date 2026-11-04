import os
import time

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from keras.applications.vgg19 import VGG19, preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from keras.models import Model

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_files")

def data_path(file_name):
    return os.path.join(DATA_DIR, file_name)

def load_and_process_image(img_path):
    loaded_img = load_img(img_path)
    array_img = img_to_array(loaded_img)
    processed_img = preprocess_input(array_img)
    expanded_processed_img = np.expand_dims(processed_img, axis=0)

    return expanded_processed_img

def deprocess_image(img):
    img = img.copy()

    img[:, :, 0] += 103.939
    img[:, :, 1] += 116.779
    img[:, :, 2] += 123.68
    img = img[:, :, ::-1]

    img = np.clip(img, 0, 255).astype("uint8")
    return img

def display_image(img):
    if len(img.shape) == 4:
        img = np.squeeze(img, axis=0)
    deprocessed_img = deprocess_image(img)

    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(deprocessed_img)
    plt.show()

def save_image(img, output_path):
    if len(img.shape) == 4:
        img = np.squeeze(img, axis=0)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.imsave(output_path, deprocess_image(img))

    return output_path

def content_cost(content_img, generated_img):
    content_activations = content_model(content_img)
    generated_activations = content_model(generated_img)

    return tf.reduce_mean(tf.square(content_activations - generated_activations))

def gram_matrix(array):
    number_of_channels = int(array.shape[-1])
    reshaped_array = tf.reshape(array, [-1, number_of_channels])
    number_of_elements = tf.shape(reshaped_array)[0]
    matrix = tf.matmul(reshaped_array, reshaped_array, transpose_a=True)

    return matrix / tf.cast(number_of_elements, tf.float32)

def style_cost(style_img, generated_img):
    overall_cost = 0

    for style_model in style_models:
        style_activations = style_model(style_img)
        generated_activations = style_model(generated_img)

        style_matrix = gram_matrix(style_activations)
        generated_matrix = gram_matrix(generated_activations)

        current_cost = tf.reduce_mean(tf.square(style_matrix - generated_matrix))
        overall_cost += current_cost * lam

    return overall_cost

def training_process(content_img_path, style_img_path, iterations = 100, alpha = 10.0, beta = 20.0):
    content_processed_img = load_and_process_image(content_img_path)
    style_processed_img = load_and_process_image(style_img_path)

    generated_img = tf.Variable(content_processed_img, dtype=tf.float32)

    optimizer = tf.optimizers.Adam(learning_rate=7.0)

    best_cost = 1e12 + 0.1
    best_img = None

    start_time = time.time()

    for i in range(iterations):
        with tf.GradientTape() as tape:
            content_costs = content_cost(content_processed_img, generated_img)
            style_costs = style_cost(style_processed_img, generated_img)
            total = content_costs * alpha + style_costs * beta

        gradients = tape.gradient(total, generated_img)
        optimizer.apply_gradients([(gradients, generated_img)])

        if total < best_cost:
            best_cost = total
            best_img = generated_img.numpy()

        print(f"Cost at iteration {i}: {best_cost}. Time elapsed: {time.time() - start_time}")

    return best_img

model = VGG19(include_top=False, weights="imagenet")
model.trainable = False

content_layer = "block5_conv2"
style_layers = ["block1_conv1", "block3_conv2", "block5_conv2"]

content_model = Model(inputs=model.inputs, outputs=model.get_layer(content_layer).output)
style_models = [Model(inputs=model.inputs, outputs=model.get_layer(style_layer).output)
                for style_layer in style_layers]

lam = 1.0 / len(style_models)

if __name__ == "__main__":
    content_path = data_path("content.jpg")
    style_path = data_path("style.jpg")
    result_path = data_path("result.png")

    best_image = training_process(content_path, style_path)

    save_image(best_image, result_path)
    print(f"Saved generated image to {result_path}")

    display_image(best_image)
