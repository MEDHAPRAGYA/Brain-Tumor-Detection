import cv2
import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from keras.layers import Rescaling
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
image_directory = 'datasets/'
no_tumor_img = os.listdir(image_directory + 'no/')
yes_tumor_img = os.listdir(image_directory + 'yes')
dataset = []
label = []

INPUT_SIZE = 64
# print(no_tumor_img)

# path = 'no0.jpg'

# print(path.split('.')[1])

for i, image_name in enumerate(no_tumor_img):
    if image_name.split('.')[1] == 'jpg':
        image = cv2.imread(image_directory + 'no/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((INPUT_SIZE, INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(0)

for i, image_name in enumerate(yes_tumor_img):
    if image_name.split('.')[1] == 'jpg':
        image = cv2.imread(image_directory + 'yes/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((INPUT_SIZE, INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(1)
# print(dataset)
# print(len(label))

dataset = np.array(dataset)
label = np.array(label)
x_train, x_test, y_train, y_test = train_test_split(dataset, label, test_size=0.2, random_state=0)

''' print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape) '''

x_train = Rescaling(scale=1./255, offset=0.0)(x_train)
x_test = Rescaling(scale=1./255, offset=0.0)(x_test)

# BUILDING THE MODEL

model = Sequential()

model.add(Conv2D(32,(3, 3), input_shape=(INPUT_SIZE, INPUT_SIZE, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32,(3, 3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64,(3, 3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))


# Binary CrossEntropy = 1, sigmoid
# Cross Entropy = 2, softmax

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=16, verbose=1, epochs=10, validation_data=(x_test, y_test), shuffle=False)
image=cv2.imread('C:\\Users\\ARPIT\\OneDrive\\Desktop\\Brain Tumor Detection\\pred\\pred0.jpg')

img=Image.fromarray(image)

img=img.resize((64,64))

img=np.array(img)

input_img=np.expand_dims(img, axis=0)

predictions = (model.predict(input_img) > 0.5).astype("int32")