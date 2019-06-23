from keras.models import load_model
from PIL import Image
from PIL import Image
import numpy as np
import os
import cv2
from keras.preprocessing.image import img_to_array

def convert_to_array(img):
    dataset = []
    im=cv2.imread(img)
    img_res=cv2.resize(im,(50,50))
    img_array = img_to_array(img_res)
    img_array=img_array/255
    dataset.append(img_array)
    return np.array(dataset)

def get_cell_name(label):
    if label==0:
        return "Uninfected"
    if label==1:
        return "Paracitized"

def predict_cell(file):
    model = load_model("malaria_Model.h5")
    print("Predicting Type of Cell Image.................................")
    ar=convert_to_array(file)
    score=model.predict(ar)
    print(score)
    label_index=np.argmax(score)
    print(label_index)
    acc=np.max(score)
    Cell=get_cell_name(label_index)
    if(Cell == "Paracitized"):
        return Cell,"The predicted Cell is a "+Cell+" and the patient is suffering from Malaria"
    else:
        return Cell,"The predicted Cell is a "+Cell+" and the patient is not suffering from Malaria"
