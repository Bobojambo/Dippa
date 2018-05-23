# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 12:16:34 2018

@author: TaitavaBobo§
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import cv2

"""
im = np.array(Image.open("RRData/images_example/1765125.jpg"), dtype=np.uint8)
# Create figure and axes
fig,ax = plt.subplots(1)
# Display the image
ax.imshow(im)

# Create a Rectangle patch
rect = patches.Rectangle((744,401),-(744-42),-(401-133),linewidth=1,edgecolor='r',facecolor='none')

# Add the patch to the Axes
ax.add_patch(rect)
plt.show()
"""


img = cv2.imread("RRData/images_example/1765125.jpg")
b,g,r = cv2.split(img)
original_image = cv2.merge([r,g,b])

# Create a Rectangle patch
rect = patches.Rectangle((744,401),-(744-42),-(401-133),linewidth=1,edgecolor='r',facecolor='none')

xmin = 42
xmax = 702
ymin = 133
ymax = 401

print("x_max, x_min, x_distance: ", xmax," ", xmin , " ", xmax-xmin)
print("y_max, y_min, y_distance: ", ymax," ", ymin , " ", ymax-ymin)
y_gap = ymax - ymin
x_gap = xmax - xmin
if x_gap > y_gap:
    pixels_to_add = int(round((x_gap - y_gap)/2))
    ymin = ymin - pixels_to_add
    ymax = ymax + pixels_to_add
else:
    pixels_to_add = int(round((y_gap - x_gap)/2))
    xmin = xmin - pixels_to_add
    xmax = xmax + pixels_to_add     

#Open cv implementation
#Extra pixels to deny black borders with the real reflected image
extra = 200
reflect_image = cv2.copyMakeBorder(img,extra,extra,extra,extra,cv2.BORDER_REFLECT_101)
b,g,r = cv2.split(reflect_image)
reflect_image_for_figure = cv2.merge([r,g,b])


img_cropped = reflect_image[(ymin + extra):(ymax + extra), (xmin + extra):(xmax + extra)]
b,g,r = cv2.split(img_cropped)
img_cropped = cv2.merge([r,g,b])

width = 128
height = 128
reSizedImage = cv2.resize(img_cropped, (width, height))

plt.subplot(221);plt.imshow(original_image) # expects distorted color
plt.subplot(222);plt.imshow(reflect_image_for_figure) # expect true color
plt.subplot(223);plt.imshow(img_cropped) # expect true color
plt.subplot(224);plt.imshow(reSizedImage) # expect true color
plt.show()

#ax.imshow(reflect)
"""
cv2.imshow("reflect image", reflect_image)
img_cropped = reflect_image[(ymin + extra):(ymax + extra), (xmin + extra):(xmax + extra)]
cv2.imshow("cropped image", img_cropped)
"""