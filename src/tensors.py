# manipuler des tenseurs en transformant des images

# thresholding
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

# Load the image file
image = tf.io.read_file("./img/etretat.jpg")
image_cerisier = tf.io.read_file("./img/cerisier.jpg")

# Decode the image
# image a des valeurs int de 0 a 255
image = tf.image.decode_jpeg(image, channels=1)  # Convert to grayscale
image_cerisier = tf.image.decode_jpeg(image_cerisier, channels=1)  # Convert to grayscale

image = tf.cast(image, tf.float32)
image_cerisier = tf.cast(image_cerisier, tf.float32)


blended_image = tf.clip_by_value(image * 0.6 + image_cerisier * 0.4, 0, 255)

blended_image = tf.cast(blended_image, tf.uint8)

# Convert the image to float32 and normalize pixel values to [0, 1]
image = tf.cast(image, tf.float32) / 255.0

# Define the threshold value
threshold = 0.5

# Apply thresholding
# tf.where(condition, true vale, false value)
thresholded_image = tf.where(image > threshold, 1.0, 0.0)

# Convert the thresholded image back to uint8
thresholded_image = tf.cast(thresholded_image * 255, tf.uint8)

# Convert the tensor to a PIL image
thresholded_pil_image = Image.fromarray(thresholded_image.numpy().squeeze())

# Display the original and thresholded images
plt.subplot(1, 2, 1)
plt.imshow(image.numpy().squeeze(), cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(thresholded_pil_image, cmap='gray')
plt.title('Thresholded Image')
plt.axis('off')

plt.tight_layout()
plt.show()

