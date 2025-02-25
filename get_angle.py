import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
from PIL import Image, ImageDraw

def get_angle_with_Hough(original_image,line_threshold = 100):
    original_image_norm = cv2.normalize(original_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    edges = cv2.Canny(original_image_norm, 100, 250)

    # Step 3: Detect lines using Hough Transform
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold = line_threshold )

    # Plot results



    # Draw detected lines on the edge-detected image
    edge_with_lines = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)  # Convert to BGR for color lines
    if lines is not None:
        for rho, theta in lines[:, 0]:  # Iterate over detected lines
            if (np.rad2deg(theta) >= 25 and np.rad2deg(theta) <= 70) or (np.rad2deg(theta) >= 110 and np.rad2deg(theta) <= 155):
            
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                cv2.line(edge_with_lines, (x1, y1), (x2, y2), (0, 0, 255), 2)
                print(f'This is the estimated angle by Hough Transformation estimation: {np.rad2deg(theta)}°')
                break
                

    else:
        return 0,0
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    axes[0].imshow(original_image, cmap='gray')
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    axes[1].imshow(edge_with_lines)
    axes[1].set_title("Hough Transform on Canny Original Image")
    axes[1].axis("off")

    plt.show()

    if theta > 90:
        return rho, theta-90
    else:
        return rho, theta






def get_angle_by_sampling(image,n=800):
    from_ = 200
    ref_col = image[from_:,10]
    till = np.argmax(ref_col)

    angles = []
    save_for_plotting = []
    for idx in range(n):
        
        a1,b1 = np.random.choice(np.arange(from_, from_+till+1),size=2,replace=False)
    
        temp_row = image[a1,5:].copy()  #Adjust Threshold.....
        temp_row2 = image[b1,5:].copy()
        if len(np.nonzero(temp_row)[0])>0 and len(np.nonzero(temp_row2)[0])>0:
            a0 = np.nonzero(temp_row)[0][0] # It is already binerized! no need to threshold
        
             # Adjust Threshold.....
        
            b0 = np.nonzero(temp_row2)[0][0] # It is already binerized! no need to threshold
        
            if a0 > b0:
                c = a0-b0
                d = b1-a1
            else:
                c = b0-a0
                d = a1-b1

            alpha = math.degrees(math.sin(c/d))    
            if alpha > 1 and a0 < len(temp_row)/2 and b0 < len(temp_row2)/2:
                angles.append(alpha)

            
            save_for_plotting = b0,b1

    return angles,save_for_plotting


# Here the functions get called ----------------------- you can change xxx.png/jpg here... ----------------------------

original_image = cv2.imread('images/image_01.png',cv2.IMREAD_GRAYSCALE).astype(float)

temp_threshold =100
while temp_threshold>0:
    rho,theta = get_angle_with_Hough(original_image,line_threshold=temp_threshold)
    if theta == 0:
        temp_threshold = temp_threshold-3
    else:
        break

original_image[original_image<10] = 0
original_image[original_image>=10] = 255

angles,point = get_angle_by_sampling(original_image)

angles = np.array(angles)

angles = 90 - angles # changing the reference angle...

bins = 60
counts, bin_edges = np.histogram(angles, bins=bins)

# Find the bin with the highest frequency
max_bin_index = np.argmax(counts)

# Get the edges of this bin
bin_start = bin_edges[max_bin_index]
bin_end = bin_edges[max_bin_index + 1]

# Filter angles that fall into this bin
angles_in_max_bin = angles[(angles >= bin_start) & (angles < bin_end)]

# Find the most frequent angle in this bin
unique_in_bin, counts_in_bin = np.unique(angles_in_max_bin, return_counts=True)
max_count_index_in_bin = np.argmax(counts_in_bin)
most_frequent_angle_in_bin = unique_in_bin[max_count_index_in_bin]

# Print the result
print(f'This is the estimated angle by Monte Carlo estimation: {most_frequent_angle_in_bin}°')

# Plot histogram
plt.hist(angles, bins=bins, edgecolor='black')
plt.axvline(most_frequent_angle_in_bin, color='red', linestyle='dashed', label=f"Most Frequent: {most_frequent_angle_in_bin}°")
plt.legend()
plt.show()


from PIL import Image, ImageDraw



# Convert angle to radians
angle_rad = np.deg2rad(most_frequent_angle_in_bin)

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
image = Image.new('RGB', original_image.shape, color='white')
draw = ImageDraw.Draw(image)

# Draw the line
draw.line([x_start, y_start, x_end, y_end], fill='red', width=3)

# Display the image
plt.imshow(original_image)  # Display the original image
plt.imshow(image, alpha=0.8)  # Overlay the second image with transparency
plt.axis('off')  # Turn off axis
plt.show()



'''TODO: A mi képeinken a balfelső sarokba van az a kocka, az utántól kell samplelni!'''