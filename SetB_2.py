import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load a 256-level grayscale image
img = cv2.imread("grayscale_image.jpg", cv2.IMREAD_GRAYSCALE)


# Define the power-law (gamma) transformation function
def power_law_transform(image, gamma):
    # Normalize pixel values to the range [0, 1]
    normalized_img = image / 255.0
    # Apply the power-law transformation
    transformed_img = np.power(normalized_img, gamma)
    # Scale back to [0, 255] and convert to unsigned 8-bit integer
    transformed_img = np.uint8(transformed_img * 255)
    return transformed_img


# Apply power-law transformation with gamma > 1 (e.g., 2.0)
gamma = 3.0
output_img = power_law_transform(img, gamma)

# Display the input and output images
plt.figure(figsize=(10, 5))

# Input image
plt.subplot(1, 2, 1)
plt.imshow(img, cmap="gray")
plt.title("Original Image")
plt.axis("off")

# Output image (after transformation)
plt.subplot(1, 2, 2)
plt.imshow(output_img, cmap="gray")
plt.title("Output Image (Gamma = {})".format(gamma))
plt.axis("off")

plt.show()

# Plot the histograms of the original and transformed images
plt.figure(figsize=(10, 5))

# Histogram of the original image
plt.subplot(1, 2, 1)
plt.hist(img.ravel(), bins=256, range=[0, 256])
plt.title("Histogram of Original Image")

# Histogram of the transformed image
plt.subplot(1, 2, 2)
plt.hist(output_img.ravel(), bins=256, range=[0, 256])
plt.title("Histogram of Transformed Image")

plt.show()
