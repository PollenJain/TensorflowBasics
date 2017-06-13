import cv2
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

data = input_data.read_data_sets("data/MNIST/", one_hot=True)
images = data.test.images[0:2] # 2 images # (2,784)
# destination = np.empty(np.array([28,28])*images.shape[0])
# destination = np.concatenate(images[:,])
# destination = np.empty((28,28))

# for image in images:
image1 = images[0].reshape((28,28))
image2 = images[1].reshape((28,28))
destination = np.concatenate(images) # (1568,)

print(destination.shape)

dest = np.concatenate((image1,image2), axis=1)
print(dest.shape)

cv2.imshow('dest', dest)
cv2.waitKey(0)
cv2.destroyAllWindows()
disp = images[0].reshape((28,28))
disp1 = images[0].reshape((28,28))

for image in images[1:]:
    disp = np.concatenate((disp,image.reshape((28,28))))
    disp1 = np.concatenate((disp1,image.reshape((28,28))),axis=1)

print(disp.shape)
# cv2.imshow('disp', disp)
cv2.imshow('disp1', disp1)
cv2.waitKey(0)
cv2.destroyAllWindows()
