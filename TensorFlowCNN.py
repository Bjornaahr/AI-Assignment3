#%%
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import numpy as np
from tensorflow.keras import datasets, layers, models
from skimage.feature import hog
from skimage import data, exposure
from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt
from PIL import Image


#%%
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0


#%%
radius = 3
n_points = 8 * radius

lbplist = []
for i in range(len(train_images)):
    lbp = local_binary_pattern(train_images[i][:,:,1], n_points, radius)
    lbplist.append(lbp)
lbplist = np.array(lbplist)

#%%
lbplist *= 1/lbplist.max()

#%%
imagesHoG = []
for i in range(len(train_images)):
    # Made size of gradients 8x8 as 4x4 seemed too small, yet 16x16 would have been too few axis
    fd, hog_image = hog(train_images[i], orientations=8, pixels_per_cell=(8, 8),
                        cells_per_block=(1, 1), visualize=True, multichannel=True)
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    imagesHoG.append(hog_image_rescaled)
imagesHoG = np.array(imagesHoG)

#%%
imagesGrey = []
for i in range(len(train_images)):
    imagesGrey.append(train_images[i][:,:,1])
imagesGrey = np.array(imagesGrey)

#%%
imagesHLG = np.stack((imagesHoG, lbplist, imagesGrey), axis=-1)

#%%
radius = 3
n_points = 8 * radius

lbplistTest = []
for i in range(len(test_images)):
    lbp = local_binary_pattern(test_images[i][:,:,1], n_points, radius)
    lbplistTest.append(lbp)
lbplistTest = np.array(lbplistTest)

#%%
lbplistTest *= 1/lbplistTest.max()


#%%
imagesHoGTest = []
for i in range(len(test_images)):
    # Made size of gradients 8x8 as 4x4 seemed too small, yet 16x16 would have been too few axis
    fd, hog_image = hog(test_images[i], orientations=8, pixels_per_cell=(8, 8),
                        cells_per_block=(1, 1), visualize=True, multichannel=True)
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    imagesHoGTest.append(hog_image_rescaled)
imagesHoGTest = np.array(imagesHoGTest)

#%%
imagesGreyTest = []
for i in range(len(test_images)):
    imagesGreyTest.append(test_images[i][:,:,1])
imagesGreyTest = np.array(imagesGreyTest)
#%%
#Combining all layers
imagesHLGTest = np.stack((imagesHoGTest, lbplistTest, imagesGreyTest), axis=-1)



#%%
print("HLG: ", imagesHLGTest[1].shape)
print("RAW: ", test_images.shape)

#%%
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.summary()


model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(64, activation='relu'))





#Last layer
model.add(layers.Dense(10, activation='softmax'))

model.summary()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(imagesHLG, train_labels, epochs=10, 
                    validation_data=(imagesHLGTest, test_labels))




test_loss, test_acc = model.evaluate(imagesHLGTest,  test_labels, verbose=2)
print(test_acc)


# %%
