import cv2
import numpy as np
#Load the Gray scale Image
gray_image = cv2.imread(r"C:\CV\cat.jpeg",cv2.IMREAD_GRAYSCALE)
#Convert Grayscale image to BGR (3-Channel) for Colorization
color_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
#Define a basic Colorization lookup table
color_lookup_table=np.zeros((256,1,3),dtype=np.uint8)
for i in range(256):
    color_lookup_table[i,0,0]=i
    color_lookup_table[i,0,1]=127
    color_lookup_table[i,0,2]=255-i
#Apply the colorization Lookup table to the Grayscale image
colorized_image=cv2.LUT(color_image,color_lookup_table)
#Display the Original Grayscale image and the colorized image
cv2.imshow('Grayscale Image',gray_image)
cv2.imshow('Colorized Image',colorized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
