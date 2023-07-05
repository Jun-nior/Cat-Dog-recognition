import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten
from keras.models import load_model

#generators/load data to train
train_ds = keras.utils.image_dataset_from_directory(
    directory = 'C:/Trung Main/HEHE/Second Deep CNN/train',
    labels='inferred',
    label_mode='int',
    batch_size=32,
    image_size=(64,64),
    seed=42
)

validation_ds = keras.utils.image_dataset_from_directory(
    directory = 'C:/Trung Main/HEHE/Second Deep CNN/test',
    labels='inferred',
    label_mode='int',
    batch_size=32,
    image_size=(64,64),
    seed=42
)

#console: found 20000 files belonging to 2 classes and 5000 files belonging to 2 classes

# Normalize
train_ds=train_ds.take(70)
validation_ds=validation_ds.take(20)
def process(image,label):
    image = tf.cast(image/255. ,tf.float32)
    return image,label

train_ds = train_ds.map(process)
validation_ds = validation_ds.map(process)

# create CNN model

# model = Sequential()

# model.add(Conv2D(32,kernel_size=(3,3),padding='valid',activation='relu',input_shape=(64,64,3)))
# model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='valid'))

# model.add(Conv2D(64,kernel_size=(3,3),padding='valid',activation='relu'))
# model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='valid'))

# model.add(Conv2D(128,kernel_size=(3,3),padding='valid',activation='relu'))
# model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='valid'))

# model.add(Flatten())

# model.add(Dense(128,activation='relu'))
# model.add(Dense(64,activation='relu'))
# model.add(Dense(1,activation='sigmoid'))
# # model.summary()

# model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

# early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)
# model_checkpoint = tf.keras.callbacks.ModelCheckpoint('C:\Trung Main\HEHE\Second Deep CNN', monitor='val_loss', save_best_only=True)

# model.fit(train_ds,epochs=1000,validation_data=validation_ds,callbacks=[early_stopping,model_checkpoint])
# model.save('image2_classifier.model')

model=load_model('C:\Trung Main\HEHE\Second Deep CNN\image2_classifier.model')

# test with input

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

img = cv.imread('C:\Trung Main\HEHE\Second Deep CNN\dog.10120.jpg')
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

resized_img=cv.resize(img,(64,64))
plt.imshow(resized_img, cmap=plt.cm.binary)
# plt.show()
test_input=resized_img.reshape((1,64,64,3))
prediction= model.predict(test_input)
if prediction==1:
    print("Dog")
else :print("Cat")

