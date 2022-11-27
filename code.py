import numpy as np
import pandas as pd

import os

import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tensorflow.keras import models, layers

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

imgs_path = "C:\Autonise_chat_bot\Traffic sign project\Train"
data_list = []
labels_list = []
classes_list = 43 # there are 43 sign-classes in total for detecting
for i in range(classes_list):
    i_path = os.path.join(imgs_path, str(i)) #0-42
    for img in os.listdir(i_path):
        im = Image.open(i_path +'/'+ img)
        im = im.resize((32,32))
        im = np.array(im)
        data_list.append(im)
        labels_list.append(i)
data = np.array(data_list)
labels = np.array(labels_list)

data

# Visualizing Dataset

plt.figure(figsize = (12,12))

for i in range(5) :
    plt.subplot(1, 5, i+1)
    plt.imshow(data[i], cmap='gray')

plt.show()

# preparing the data
from tensorflow.keras.utils import to_categorical
def prep_dataset(X,y):
    X_prep = X.astype('float32')
    y_prep = to_categorical(np.array(y))
    return (X_prep, y_prep)

X, y = prep_dataset(data,labels)

# splitting for training and validating dataset

X_train, X_val, y_train, y_val = train_test_split(X,y, test_size=0.2, shuffle=True,stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_val,y_val, test_size=0.5, shuffle=True)

# model creation 
# CNN

model = models.Sequential() #Sequential Model

#ConvLayer(32 filters) + MaxPooling + BatchNormalization + Dropout
model.add(layers.Conv2D(filters=32,kernel_size=3,activation='relu',padding='same',input_shape=X.shape[1:]))
model.add(layers.MaxPool2D(strides=2))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.3))

#ConvLayer(128 filters) + MaxPooling + BatchNormalization + Dropout
model.add(layers.Conv2D(filters=128,kernel_size=3,activation='relu',padding='same'))
model.add(layers.MaxPool2D(strides=2))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.3))

#ConvLayer(512 filters) + Dropout + ConvLayer(512 filters) + MaxPooling + BatchNormalization
model.add(layers.Conv2D(filters=512,kernel_size=3,activation='relu',padding='same'))
model.add(layers.Dropout(0.3))
model.add(layers.Conv2D(filters=512,kernel_size=3,activation='relu',padding='same'))
model.add(layers.MaxPool2D(strides=2))
model.add(layers.BatchNormalization())

#Flatten
model.add(layers.Flatten())

#2 Dense layers with 4000 hidden units
model.add(layers.Dense(4000,activation='relu'))
model.add(layers.Dense(4000,activation='relu'))

#Dense layer with 1000 hidden units
model.add(layers.Dense(1000,activation='relu'))

#Softmax layer for output
model.add(layers.Dense(43,activation='softmax'))

model.summary()

# compiling and fitting of data

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

history= model.fit(X_train,y_train,epochs=10,batch_size=64,validation_data=(X_val,y_val))

# prediction and evaluation 

y_test = np.argmax(y_test,axis=1)
y_pred= model.predict(X_test)

y_pred = np.argmax(y_pred, axis=1)

print('- Train dataset Acuracy achieved: {:.2f}%\n-Accuracy by model was: {:.2f}%\n-Accuracy by validation was: {:.2f}%'.
      format(accuracy_score(y_test,y_pred)*100,(history.history['accuracy'][-1])*100,(history.history['val_accuracy'][-1])*100))

plt.figure(0)
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()
plt.figure(1)
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

model.save("my_model.h5")

from sklearn.metrics import accuracy_score
y_test = pd.read_csv('Test.csv')
labels = y_test["ClassId"].values
imgs = y_test["Path"].values
data=[]
path='C:\Autonise_chat_bot\Traffic sign project'
for img in imgs:
    image = Image.open(path +'\\'+ img)
    image = image.resize((30,30))
    data.append(np.array(image))




model.save('traffic_classifier.h5')


