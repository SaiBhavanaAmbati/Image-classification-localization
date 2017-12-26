from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
import xml.etree.ElementTree as ET
import pandas as pd
from PIL import Image
import numpy
import os
import imageio
import numpy as np
#import opencv
base='/home/bhavana/Documents/ai/cnn/VOCdevkit/VOC2010/Annotations/'
base1='/home/bhavana/Documents/ai/cnn/VOCdevkit/VOC2010/JPEGImages/'
x_all=[]
y_all=[]
table={}
table['person']=0
table['bird']=1
table['cat']=2
table['cow']=3
table['dog']=4
table['horse']=5
table['sheep']=6
table['aeroplane']=7
table['bicycle']=8
table['boat']=9
table['bus']=10
table['car']=11
table['motorbike']=12
table['train']=13
table['bottle']=14
table['chair']=15
table['diningtable']=16
table['pottedplant']=17
table['sofa']=18
table['tvmonitor']=19
X_train=[]
Y_train=[]
X_test=[]
Y_test=[]
count=0
def xml2df(xml_data):
    root = ET.XML(xml_data) # element tree
    all_records = {}
    for i, child in enumerate(root):
        i=0
        for subchild in child:
            all_records[subchild.tag] = subchild.text
            i=i+1
        if(i==0):
            all_records[child.tag] = child.text
    y=np.zeros(20)
    index=all_records["name"]
    index2=table[index]
    y[index2]=1
    # y_all.append(y)
    W = Image.open(base1+all_records["filename"])
    W = W.resize((200,200))
    W.save(base1+all_records["filename"])
    im = imageio.imread(base1+all_records["filename"])
    global count
    if count<1000:
        X_train.append(im)
        Y_train.append(y)
    else:
        X_test.append(im)
        Y_test.append(y)
    count=count+1
    # x_all.append(im)
#xml2df(xml_data)
k=0
for filename in os.listdir(base):
    if k==1500:
        break
    k=k+1
    #print base+filename 
    xml_data = open(base+filename).read()
    xml2df(xml_data)

X_train= np.array(X_train)
Y_train=np.array(Y_train)
X_test= np.array(X_test)
Y_test=np.array(Y_test)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
print Y_train.shape
model = Sequential()
model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(200,200,3)))
model.add(Convolution2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(20, activation='softmax'))
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train, Y_train, 
          batch_size=32, epochs=50, verbose=1, validation_split=0.1)

loss, accuracy = model.evaluate(X_test, Y_test, verbose=0)
print('Accuracy: %f' % (accuracy*100))
# transfer learning
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.layers import Input, Flatten, Dense
from keras.models import Model
model_vgg16_conv = VGG16(weights='imagenet', include_top=False)
model_vgg16_conv.summary()
input1 = Input(shape=(200,200,3),name = 'image_input')
output_vgg16_conv = model_vgg16_conv(input1)
 x = Flatten(name='flatten')(output_vgg16_conv)
x = Dense(64, activation='relu', name='fc1')(x)
x = Dense(64, activation='relu', name='fc2')(x)
x = Dense(20, activation='softmax', name='predictions')(x)
my_model = Model(input=input1, output=x)
my_model.summary()
model.compile(loss='mse',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_test,Y_test, batch_size=32, epochs=50,verbose=1, validation_split=0.1)
