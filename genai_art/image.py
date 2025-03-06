import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import imageio

def get_image(img_path, nSize=256):
    img_ext = 'png'
    if img_ext == 'png':
        # Read the png file
        image_string = tf.io.read_file(img_path)
        # Decode the image
        image = tf.image.decode_png(image_string, channels=1)  # Adjust 'channels' if needed
    # Resize the image
    image_resized = tf.image.resize(image, [nSize, nSize])
    image_resized2 = image_resized.numpy().reshape(nSize, nSize)
    return image_resized2

def display(image):
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()

def display_rgb(image):
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def clip_black_white(image, threshold=128):
    return np.where(image > threshold, 255, 0)

def save_animation(frames, filename, fps=5):
    frames_uint8 = [np.clip(frame * 255, 0, 255).astype(np.uint8) for frame in frames]
    imageio.mimsave(filename, frames_uint8, fps=fps)
