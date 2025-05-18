import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import numpy as np

# Load the pre-trained model from TensorFlow Hub
hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

def load_image(image_file, image_size=(256, 256)):
    # Open and resize image
    image = Image.open(image_file).convert('RGB')
    image = image.resize(image_size)
    
    # Normalize to [0, 1] and convert to float32
    img = np.array(image).astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return tf.constant(img, dtype=tf.float32)

def stylize_image(content_image_file, style_image_file):
    # Load images
    content_image = load_image(content_image_file, image_size=(384, 384))
    style_image = load_image(style_image_file, image_size=(256, 256))

    # Apply style transfer
    stylized_image = hub_model(content_image, style_image)

    return stylized_image  # Returns tuple (stylized_image,)
