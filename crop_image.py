import cv2
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np



main_img = cv2.imread("image.png", cv2.IMREAD_GRAYSCALE)  # Load main image in grayscale
kernel = main_img[61:79,25:43].copy()

# Apply template matching
result = cv2.matchTemplate(main_img, kernel, cv2.TM_CCOEFF_NORMED)

# Get the best match location
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

# Draw a rectangle around the detected template
h, w = kernel.shape  # Get dimensions of the template
top_left = max_loc  # Best match location
bottom_right = (top_left[0] + w, top_left[1] + h)

cv2.rectangle(main_img, top_left, bottom_right, (255, 255, 255), 2)  # Draw rectangle

# Show the result
cv2.imshow("Detected Shape", main_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# original_image = cv2.imread('image.png',cv2.IMREAD_GRAYSCALE).astype(float)
# kernel = original_image[51:79,15:43].copy()
# kernel[kernel < 5] = -100

# n,m = original_image.shape

# filtered = signal.convolve(original_image[:int(n/5),:int(m/3)], kernel, mode='full',method ='direct') 

# ind = np.unravel_index(np.argmax(filtered, axis=None), filtered.shape)

# dummy_img = np.zeros([n,m])
# dummy_img[ind] = 1

# plt.imshow(original_image)
# plt.show()


# plt.imshow(filtered)
# plt.show()



# plt.imshow(filtered)
# plt.show()

# print(f'This is the index of maximum, after convoltuion: {ind}')
# print(f'This is the [corrected] index of maximum, after convoltuion: x:{ind[1]-8-3}, y:{ind[0]-8-3}')



