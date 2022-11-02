# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from fastapi import FastAPI, File, UploadFile
import uvicorn
import Predict
import asyncio
import numpy as np
from io import BytesIO
from PIL import Image
from tensorflow import keras
import pickle

import nest_asyncio
nest_asyncio.apply()

app = FastAPI()

@app.get('/')
async def index():
    return {'Hi'}

@app.get('/Name')
async def get_name_of_user(name: str):
    return {'hi:',name}

@app.post('/Predict')
async def get_image(file: UploadFile = File(...)):
    data = []
    
    contents = await file.read()
    image = Image.open(BytesIO(contents))
    image = image.resize((30,30))
    image = np.array(image)
    
    data.append(image)
    model = keras.models.load_model('C:/Users/vujvalsai/.spyder-py3/FastAPI/my_model.h5')
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    data = np.array(data)
    #print(data.shape)
    
    n_image = 1
    image = image.reshape(n_image,30,30,3)
    
    result = np.argmax(model.predict(image))
    
    return{'Image class is :', int(result)}
  
    
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)