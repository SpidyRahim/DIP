import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load a 256-level grayscale image
img = cv2.imread("grayscale_image.jpg", cv2.IMREAD_GRAYSCALE)

# Check if the image was loaded successfully
if img is None:
    print("Error: Image not loaded. Please check the file path.")
    exit()


# Function to find the threshold where tf1 is approximately equal to tf2
def find_threshold(image):
    # Flatten the image to a 1D array for easier manipulation
    pixel_values = image.ravel()

    total_pixels = len(pixel_values)

    for threshold in range(256):  # Iterate through possible threshold values
        tf1 = np.sum(pixel_values <= threshold)
        tf2 = np.sum(pixel_values > threshold)

        # Check if tf1 is approximately equal to tf2
        if abs(tf1 - tf2) <= total_pixels * 0.01:  # 1% tolerance
            return threshold

    return 128  # If no threshold is found, return a default value (e.g., 128)


# Find the appropriate threshold
threshold = find_threshold(img)

# Generate the binary image using the found threshold
_, binary_img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)

# Display the original grayscale image and the binary image
plt.figure(figsize=(10, 5))

# Original grayscale image
plt.subplot(1, 2, 1)
plt.imshow(img, cmap="gray")
plt.title("Original Grayscale Image")
plt.axis("off")

# Binary image
plt.subplot(1, 2, 2)
plt.imshow(binary_img, cmap="gray")
plt.title(f"Binary Image (Threshold = {threshold})")
plt.axis("off")

plt.show()

# Print the threshold value
print(f"The threshold where tf1 is approximately equal to tf2: {threshold}")
