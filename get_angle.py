import numpy as np

import cv2
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft,fftshift


original_image = cv2.imread('ultrasound.jpg',cv2.IMREAD_GRAYSCALE).astype(float)

original_image[original_image<10] = 0
original_image[original_image>=10] = 255

original_image_norm = cv2.normalize(original_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

edges = cv2.Canny(original_image_norm, 50, 150)


# Step 3: Detect lines using Hough Transform
lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)

# Plot results
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

axes[0].imshow(original_image, cmap='gray')
axes[0].set_title("Original Image")
axes[0].axis("off")


# Draw detected lines on the edge-detected image
edge_with_lines = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)  # Convert to BGR for color lines
if lines is not None:
    for rho, theta in lines[:, 0]:  # Iterate over detected lines
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(edge_with_lines, (x1, y1), (x2, y2), (0, 0, 255), 2)

axes[1].imshow(edge_with_lines)
axes[1].set_title("Hough Transform on Canny Original Image")
axes[1].axis("off")

plt.show()


'''TODO: Get the angles out of rho, theta from line, define boundary conditions for them, 
if more lines found (if no lines found, thickening of objects might help. Also, thresholds need to be adjusted.)'''