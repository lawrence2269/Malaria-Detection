#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 15:48:34 2019

@author: lawrence
"""

import matplotlib.pyplot as plt
import numpy as np
import pickle
import glob
import gc
import cv2

# Path array of infected cell images
infected_cells=glob.glob("/Users/lawrence/Documents/Active Learning Project - 2/cell_images/Parasitized/*.png")
uninfected_cells=glob.glob("/Users/lawrence/Documents/Active Learning Project - 2/cell_images/Uninfected/*.png")

for i in range(0,1000):
    gc.collect()

plt.figure(figsize=(15,15))
for i in range(1,6):
    plt.subplot(1,5,i)
    ran=np.random.randint(100)
    plt.imshow(cv2.imread(infected_cells[ran]))
    plt.title('Infected cell')

plt.figure(figsize=(15,15))
for i in range(1,6):
    plt.subplot(1,5,i)
    ran=np.random.randint(100)
    plt.imshow(cv2.imread(uninfected_cells[ran]))
    plt.title('Uninfected cell')
    
pickle_in = open("model_metrics.pickle","rb")
alp = pickle.load(pickle_in)

# summarize history for accuracy
plt.plot(alp['acc'])
plt.plot(alp['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='lower right')
plt.show()

# summarize history for loss
plt.plot(alp['loss'])
plt.plot(alp['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper right')
plt.show()