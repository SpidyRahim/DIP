import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load a 256-level grayscale image
img = cv2.imread("grayscale_image.jpg", cv2.IMREAD_GRAYSCALE)

# Check if the image was loaded successfully
if img is None:
    print("Error: Image not loaded. Please check the file path.")
    exit()


# Define the logarithmic transformation function
def log_transform(image):
    # Convert image to float for the transformation
    image_float = np.float32(image)

    # Apply the log transform s = c * log(1 + r)
    c = 255 / np.log(1 + np.max(image_float))  # Scaling factor
    log_image = c * np.log(1 + image_float)

    # Convert the image back to unsigned 8-bit integer
    log_image = np.uint8(log_image)

    return log_image


# Apply the logarithmic transformation
output_img = log_transform(img)

# Display the input and output images
plt.figure(figsize=(10, 5))

# Original image
plt.subplot(1, 2, 1)
plt.imshow(img, cmap="gray")
plt.title("Original Image")
plt.axis("off")

# Output image (after logarithmic transformation)
plt.subplot(1, 2, 2)
plt.imshow(output_img, cmap="gray")
plt.title("Output Image (Log Transform)")
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
plt.title("Histogram of Transformed Image (Increasing)")

plt.show()
