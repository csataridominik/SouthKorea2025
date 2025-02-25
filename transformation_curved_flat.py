from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
import cv2
import math as m

# Step 1: Load the image
image_path = "C:\\Users\\buvr_\\Documents\\BUVR 2025.1\\transforming recordings\\samples\\sample_curved_cropped_01.png"  # Replace with your image file path
image = Image.open(image_path)

# Step 2: Convert to grayscale
gray_image = image.convert("L")  # "L" mode is for grayscale

# Step 3: Convert to NumPy array
gray_array = np.array(gray_image)

a = b = False
column = []
for i in range(1230):
    if gray_array[2][i] > 10:
        a = True
    else:
        a = False
    if a != b and a == False:
        column.append(i)
    b = a
print('col ', column)

a = b = False
row = []
for i in range(790):
    if gray_array[i][2] > 10:
        a = True
    else:
        a = False
    if a != b and a == False:
        row.append(i)
    b = a
print('row ', row)

print(gray_array.shape)

row_cm = (row[-1]-row[1]) / (len(row)-2)
column_cm = (column[-1]-column[1]) / (len(column)-2)
height = row_cm
width = column_cm
print('row cm: ', row_cm, '; column cm: ', column_cm)

#### curved specifikus rész

print('column middle ', column[5])
'''
kovep_vege = 0
a = True
for i in range(0, len(gray_array)):
    if not a:
        b = True
        for j in range(column[4], column[5]-2):
            if gray_array[i][j] > 60 and b:
                print('megvan! ', i)
                b = False

    if gray_array[i][column[5]-2] < 10 and a:
        print('kozep vege ', i)
        kozep_vege = i
        a = False
'''


gray_row_sums = np.sum(gray_array, axis = 1)
gray_row_id = np.where(gray_row_sums<1500)[0][0]


temp = gray_array[gray_row_id:,6:].copy().astype(float) # kozep_vege eredetileg 10 volt
temp[temp>200] = 0
temp[temp<5] = 0
temp[:100,:100] = 0
temp[:100,-200:] = 0
row_sums = np.sum(temp, axis = 1)

row_id = np.where(row_sums>200)[0][0]
column_ids = sc.signal.find_peaks(temp[row_id,:],distance=100)[0]
cln = column_ids.copy()
for i in range(len(column_ids)):
    # kiszűrjük azokat a pontokat, amiket nem a megfelelő tartományban találtunk -> finomítani lehetne hogy csak 2 maradjon
    if abs(column_ids[i]) > len(gray_array[0])*2/3:
        cln = np.delete(column_ids, i)    


column_sums = np.sum(temp, axis = 0)
### 
#plt.figure
#plt.plot(column_sums)
#plt.show()

### 
#plt.figure
#plt.plot(np.diff(row_sums))
#plt.show()


### csillaggal jelölni felső sarkokat
#cln1 = cln.copy() ##
#plt.imshow(temp>0)
#plt.plot(cln,row_id*np.ones(cln.shape),'r*')
#plt.show()

#print(cln)



gray_array_float = gray_array.astype(float)
first = gray_array[:,1]
conv_results = np.asarray([np.matmul(first.T,  gray_array_float[:,col]) for col in range(gray_array_float.shape[1])])
### 
#plt.plot(conv_results)
#plt.show()

### 
#plt.imshow(gray_array>1)
#plt.show()

'''

img = cv2.imread(image_path)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(gray, 3, 150, apertureSize=3)

lines = cv2.HoughLines(edges, 1, np.pi/180, 5)

# The below for loop runs till r and theta values
# are in the range of the 2d array
for r_theta in lines:
    arr = np.array(r_theta[0], dtype=np.float64)
    r, theta = arr
    # Stores the value of cos(theta) in a
    a = np.cos(theta)

    # Stores the value of sin(theta) in b
    b = np.sin(theta)

    # x0 stores the value rcos(theta)
    x0 = a*r

    # y0 stores the value rsin(theta)
    y0 = b*r

    # x1 stores the rounded off value of (rcos(theta)-1000sin(theta))
    x1 = int(x0 + 1000*(-b))

    # y1 stores the rounded off value of (rsin(theta)+1000cos(theta))
    y1 = int(y0 + 1000*(a))

    # x2 stores the rounded off value of (rcos(theta)+1000sin(theta))
    x2 = int(x0 - 1000*(-b))

    # y2 stores the rounded off value of (rsin(theta)-1000cos(theta))
    y2 = int(y0 - 1000*(a))

    # cv2.line draws a line in img from the point(x1,y1) to (x2,y2).
    # (0,0,255) denotes the colour of the line to be
    # drawn. In this case, it is red.
    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

# All the changes made in the input image are finally
# written on a new image houghlines.jpg
cv2.imwrite('linesDetected.jpg', img)
'''

'''
lines_list =[]
lines = cv2.HoughLinesP(
            edges, # Input edge image
            1, # Distance resolution in pixels
            np.pi/180, # Angle resolution in radians
            threshold=10, # Min number of votes for valid line
            minLineLength=5, # Min allowed length of line
            maxLineGap=1000 # Max allowed gap between line for joining them
            )

# Iterate over points
for points in lines:
      # Extracted points nested in the list
    x1,y1,x2,y2=points[0]
    # Draw the lines joing the points
    # On the original image
    cv2.line(image,(x1,y1),(x2,y2),(0,255,0),2)
    # Maintain a simples lookup list for points
    lines_list.append([(x1,y1),(x2,y2)])
    
# Save the result image
cv2.imwrite('detectedLines.png',image)
'''



#### dolgozzunk 55 fokkal -> transzformáció

theta = 84/180*np.pi

alpha_d = 42/180*np.pi
ratio = height / width
print(cln[0], ' ', cln[1])
d = abs(cln[0] - cln[1])/2
print('d ', d)
d_cm = d/column_cm
print('d cm ', d_cm)
#offset_cm = d_cm/np.sin(theta/2)#m.floor(d_cm/np.sin(theta/2))
alpha_real = m.atan(m.tan(alpha_d)*ratio)
print('alpha: ', alpha_d, ' alpha real: ', alpha_real)
offset_cm = d_cm/np.sin(alpha_real)
print('offset cm ', offset_cm)
offset = offset_cm*row_cm
print('offset ', offset)
middle_column = m.floor(cln[0] + d)
print(middle_column)



# plt.figure
# #plt.plot(cln,row_id*np.ones(cln.shape),'r*')
# arr = temp[:, middle_column]
# plt.plot(arr)
# plt.show()

# temp[:, middle_column] = 255
# plt.imshow(temp)
# plt.show()

last = 0
for i in range(len(temp)):
    if temp[i][middle_column] > 0:
        last = i


print(last)
print(temp.shape)
#temp[last][middle_column] = 255
# plt.imshow(temp)
# plt.show()
r = last - offset
r_cm = r/height
print('r: ', r, '; offset: ', offset)
print('r_cm: ', r_cm, '; offset_cm: ', offset_cm)

#### ÁTVÁLTANI CM-BE PIXELBŐL

r_mm = r_cm*10
offset_mm = offset_cm*10

print('mm ', r_mm, offset_mm)

alpha_deg = alpha_real*180/m.pi
alpha = round(alpha_deg)
print('degrees ', alpha_deg, alpha)

# Define row and column ranges
rows = np.arange(m.floor(offset_mm), m.floor(r_mm))  # 23 to 788 inclusive
cols = np.arange(m.floor(-alpha*2), m.floor(alpha*2))   # 27 to 81 inclusive

# Create meshgrid
R, Th = np.meshgrid(cols, rows)
Intensity = R.copy()

# Print the shape of the meshgrid
print("X shape:", R.shape)
print("Y shape:", Th.shape)

print(R)
print(Th)

# R és Th értékekből x,y kiszámolása, behelyezni egy új meshgridbe r th koordinátákkal a képből x y szerinti intenzitást

#for i in range(len(R)):
#    for j in range(len(R[0])):

Y = np.cos(Th)*R
X = np.sin(Th)*R

print(X)
print(Y)

'''CENTERPONT!!!!!'''
'''TRANSZFORMÁLÁS POLÁRBÓL XY-BA HIBÁS?'''
'''XY CM-BEN VAGY PIXELBEN VAN?'''

def trilinear_interpolation(X, Y, temp):   

    X_left = np.floor(X).astype(int)
    X_right = np.ceil(X).astype(int)
    Y_top = np.floor(Y).astype(int)
    Y_bottom = np.ceil(Y).astype(int)

    Intensity = np.zeros(X.shape)
    for i in range(len(Intensity)):
        for j in range(len(Intensity[0])):
            x = X[i][j]
            y = Y[i][j]
            xl = X_left[i][j]
            xr = X_right[i][j]
            yt = Y_top[i][j]
            yb = Y_bottom[i][j]
            A = temp[yt, xl]
            B = temp[yb, xl]
            C = temp[yt, xr]
            D = temp[yb, xr]
            Intensity[i][j] = (((x-xl)*C)/(xr-xl) + ((xr-x)*A)/(xr-xl)*(yb-y)/(yb-yt)) + (((x-xl)*D)/(xr-xl) + ((xr-x)*B)/(xr-xl)*(y-yt)/(yb-yt))

    return Intensity


Intensity = trilinear_interpolation(X, Y, temp)
plt.imshow(Intensity)
plt.show()


