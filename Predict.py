# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 12:59:49 2022

@author: vujvalsai
"""

from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import pickle
import numpy as np
from tensorflow import keras



def image_load(file):
    plt.imshow(file)
    data = []
    
    image = file.resize((30,30))
    image = np.array(image)
    
    print(image.shape)
    
    data.append(image)
    model = keras.models.load_model('C:/Users/vujvalsai/.spyder-py3/FastAPI/my_model.h5')
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    data = np.array(data)
    #print(data.shape)
    
    n_image = 1
    image = image.reshape(n_image,30,30,3)
    
    result = np.argmax(model.predict(image))
    print(result)
    
    return image
    

if __name__ == '__main__':
    
    a = (image_load(Image.open(r'C:\Users\vujvalsai\.spyder-py3\FastAPI\TF2.png')))
    