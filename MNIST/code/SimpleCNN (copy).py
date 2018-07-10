# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, Dropout

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
img_rows, img_cols = 28, 28
num_classes = 10

def prep_data(raw):
    y = raw[:, 0]
    out_y = keras.utils.to_categorical(y, num_classes)
    
    x = raw[:,1:]
    num_images = raw.shape[0]
    out_x = x.reshape(num_images, img_rows, img_cols, 1)
    out_x = out_x / 255
    return out_x, out_y

train_file = "../input/train.csv"
data = np.loadtxt(train_file, skiprows=1, delimiter=',')
x, y = prep_data(data)

#Specifying Model Architecture
model = Sequential()
model.add(Conv2D(24, kernel_size=(3,3), activation='relu', input_shape=(img_rows,img_cols,1)))
model.add(Dropout(0.3))
model.add(Conv2D(24, kernel_size=(3,3), activation='relu'))
model.add(Dropout(0.3))
model.add(Conv2D(24, kernel_size=(3,3), activation='relu'))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(100,activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

#Compiling the model
model.compile(loss = keras.losses.categorical_crossentropy, optimizer = 'adam', metrics = ['accuracy'])

#Fitting the model
model.fit(x,y,batch_size=100, epochs = 4, validation_split = 0.2)