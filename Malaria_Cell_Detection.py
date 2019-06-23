#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 17:49:32 2019

@author: lawrence
"""

#Importing packages to use
import cv2
import numpy as np
import glob
#import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pickle

#Model Building and Neural Network
# Keras Imports
from keras.preprocessing.image import img_to_array
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.models import Sequential

# Path array of infected cell images
infected_cells=glob.glob("/Users/lawrence/Documents/Active Learning Project - 2/cell_images/Parasitized/*.png")
uninfected_cells=glob.glob("/Users/lawrence/Documents/Active Learning Project - 2/cell_images/Uninfected/*.png")

#Dataset Preparation
dataset=[]
labels=[]

for i in infected_cells:
    img=cv2.imread(i)
    img_res=cv2.resize(img,(50,50)) #Resizing the data
    img_array = img_to_array(img_res)
    img_array=img_array/255 #normalizing the data
    dataset.append(img_array)
    labels.append(1)

for j in uninfected_cells:
    img=cv2.imread(j)
    img_res=cv2.resize(img,(50,50))
    img_array = img_to_array(img_res)
    img_array=img_array/255
    dataset.append(img_array)
    labels.append(0)

#Convert list to numpy array
images = np.array(dataset)
label_arr = np.array(labels)

#arange the indices
index = np.arange(images.shape[0])

#shuffle the indices
np.random.shuffle(index)
images = images[index]
label_arr = label_arr[index]

#spliting the training data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(images,label_arr,test_size=0.20,
                                               random_state=42)

#Convert to class labels categorical
y_train = np_utils.to_categorical(y_train, num_classes=2)
y_test = np_utils.to_categorical(y_test, num_classes=2)

model=Sequential()
model.add(Conv2D(filters=16,kernel_size=2,padding="same",activation="relu",input_shape=(50,50,3)))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=32,kernel_size=2,padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=64,kernel_size=2,padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(500,activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(2,activation="softmax"))#2 represent output layer neurons
model.summary()

#model.add(Conv2D(filters=64,kernel_size=3,padding="same",activation="relu",input_shape=(224,224,3),strides=1))
#model.add(Conv2D(filters=64,kernel_size=3,padding="same",activation="relu",strides=1))
#model.add(MaxPooling2D(pool_size=2,strides=2))

#model.add(Conv2D(filters=128,kernel_size=3,padding="same",activation="relu",strides=1))
#model.add(Conv2D(filters=128,kernel_size=3,padding="same",activation="relu",strides=1))
#model.add(MaxPooling2D(pool_size=2,strides=2))

#model.add(Conv2D(filters=256,kernel_size=3,padding="same",activation="relu",strides=1))
#model.add(Conv2D(filters=256,kernel_size=3,padding="same",activation="relu",strides=1))
#model.add(MaxPooling2D(pool_size=2,strides=2))

#model.add(Conv2D(filters=512,kernel_size=3,padding="same",activation="relu",strides=1))
#model.add(Conv2D(filters=512,kernel_size=3,padding="same",activation="relu",strides=1))
#model.add(Conv2D(filters=512,kernel_size=3,padding="same",activation="relu",strides=1))
#model.add(MaxPooling2D(pool_size=2,strides=2))

#model.add(Conv2D(filters=512,kernel_size=3,padding="same",activation="relu",strides=1))
#model.add(Conv2D(filters=512,kernel_size=3,padding="same",activation="relu",strides=1))
#model.add(Conv2D(filters=512,kernel_size=3,padding="same",activation="relu",strides=1))
#model.add(MaxPooling2D(pool_size=2,strides=2))

#model.add(Flatten())
#model.add(Dense(4096,activation="relu"))
#model.add(Dense(4096,activation="relu"))
#model.add(Dense(2,activation="softmax"))

#model.summary()

import gc 
for i in range(0,1000):
    gc.collect()
 

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

hist = model.fit(x_train,y_train,batch_size=50,epochs=10,verbose=1,validation_data=(x_test, y_test))


#model.save('malaria_Model.h5')

for i in range(0,1000):
    gc.collect()
    
alp = hist.history
pickle_out = open("model_metrics.pickle","wb")
pickle.dump(alp,pickle_out)
pickle_out.close()

pred = model.predict(x_test)

from sklearn.metrics import accuracy_score, classification_report
score = round(accuracy_score(y_test.argmax(axis=1), pred.argmax(axis=1)),2)
print(score)
report = classification_report(y_test.argmax(axis=1), pred.argmax(axis=1))
print(report)
