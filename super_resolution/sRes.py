
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time

# model for super resolution in opencv
from cv2 import dnn_superres

sr = dnn_superres.DnnSuperResImpl_create()

# read image
image = cv2.imread("Media/fman1.jpg")

# display image
plt.figure(figsize=[10,10])
plt.imshow(image[:,:,::-1], interpolation = 'bicubic');plt.axis('off');

# Define model path, if you want to use a different model then just change this path.
model_path = "models/LapSRN_x8.pb"

# Extract model name
model_name = model_path.split('/')[1].split('_')[0].lower()
# Extract model scale
model_scale = int(model_path.split('/')[1].split('_')[1].split('.')[0][1])

# display name , scale
print("model name: "+ model_name)
print("model scale: " + str(model_scale))

# Read the desired model
sr.readModel(model_path)

# we select model scale to do respective preprocessing which the model will do

# Set the desired model and scale
sr.setModel(model_name,model_scale)

# Upscale the image
Final_Img = sr.upsample(image)

print('Shape of Original Image: {} , '
      'Shape of Super Resolution Image: {}'.format(image.shape, Final_Img.shape))

# Display Image
plt.figure(figsize=[18,18])
plt.subplot(2,1,1);plt.imshow(image[:,:,::-1], interpolation = 'bicubic');plt.title("Original Image");plt.axis("off");
plt.subplot(2,1,2);plt.imshow(Final_Img[:,:,::-1], interpolation = 'bicubic');
plt.title("SR Model: {}, Scale: {}x ".format(model_name.upper(),model_scale)); plt.axis("off");
plt.show()
# Save the image
cv2.imwrite("outputs/testoutput.png", Final_Img);

