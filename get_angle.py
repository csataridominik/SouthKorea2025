import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
from PIL import Image, ImageDraw

def get_angle_with_Hough(original_image):
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


def get_angle_by_sampling(image,n=500):
    from_ = 100
    ref_col = image[from_:,10]
    till = np.argmax(ref_col)
    

    
    angles = []
    save_for_plotting = []
    for idx in range(n):
        
        a1,b1 = np.random.choice(np.arange(from_, from_+till+1),size=2,replace=False)
    
        temp_row = image[a1,5:].copy()  #Adjust Threshold.....
        a0 = np.nonzero(temp_row)[0][0] # It is already binerized! no need to threshold
        temp_row = image[b1,5:].copy() # Adjust Threshold.....
        b0 = np.nonzero(temp_row)[0][0] # It is already binerized! no need to threshold
        
        if a0 > b0:
            c = a0-b0
            d = b1-a1
        else:
            c = b0-a0
            d = a1-b1

        alpha = math.degrees(math.sin(c/d))    
        if alpha > 1 and a0 < len(temp_row)/2 and b0 < len(temp_row)/2:
            angles.append(alpha)

        if idx == 10:
            save_for_plotting = b0,b1

    return angles,save_for_plotting


original_image = cv2.imread('ultrasound.jpg',cv2.IMREAD_GRAYSCALE).astype(float)

original_image[original_image<20] = 0
original_image[original_image>=20] = 255

angles,point = get_angle_by_sampling(original_image)

angles = np.array(angles)

angles = 90 - angles # changing the reference angle...

unique, counts = np.unique(angles, return_counts=True)

# Find the index of the maximum count
max_count_index = np.argmax(counts)

# Get the most frequent item
most_frequent_item = unique[max_count_index]

print(most_frequent_item)
# original_image[original_image > 0] = 0
# original_image[y,10] = 255


plt.hist(angles, bins=100, edgecolor='black')

# Add labels and title
plt.xlabel('Angle (degrees)')
plt.ylabel('Frequency')
plt.title('Histogram of Angles')

plt.show()

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw



# Convert angle to radians
angle_rad = np.deg2rad(most_frequent_item)

# Define a length for the line (just for visualization)
line_length = 20

# Calculate the direction of the line based on the angle
dx = line_length * np.cos(angle_rad)
dy = -line_length * np.sin(angle_rad)  # negative for correct image y-direction

# End points of the line (extend the line both ways)
x_end = int(point[0] + dx)
y_end = int(point[1] + dy)

x_start = int(point[0] - dx)
y_start = int(point[1] - dy)

# Create a blank image
image = Image.new('RGB', (300, 300), color='white')
draw = ImageDraw.Draw(image)

# Draw the line
draw.line([x_start, y_start, x_end, y_end], fill='black', width=3)

# Display the image
plt.imshow(original_image)  # Display the original image
plt.imshow(image, alpha=0.5)  # Overlay the second image with transparency
plt.axis('off')  # Turn off axis
plt.show()



'''TODO: A mi képeinken a balfelső sarokba van az a kocka, az utántól kell samplelni!'''