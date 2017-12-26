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
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.layers import Input, Flatten, Dense
from keras.models import Model
from keras import backend as K
#import opencv
base='/home/bhavana/Documents/ai/cnn/VOCdevkit/VOC2010/Annotations/'
base1='/home/bhavana/Documents/ai/cnn/VOCdevkit/VOC2010/JPEGImages/'
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
print "hi"
def f1_score(y_true, y_pred):

    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))
    if c3 == 0:
        return 0
    precision = c1 / c2
    print("Precision : ",precision)
    

    
    recall = c1 / c3
    print("Recall : ",recall)

   
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score

def precision(y_true, y_pred):


    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

    if c3 == 0:
        return 0

    p = c1 / c2
    return p
    
    
def recall(y_true, y_pred):

    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

    if c3 == 0:
        return 0

    r = c1 / c3
    return r



def xml2df(xml_data):
    root = ET.XML(xml_data)
    all_records = {}
    store=[]
    for i, child in enumerate(root):
        i=0
        if (child.tag) not in store :
            store.append(child.tag)
            for subchild in child:
                box = {}
                j=0
                for subsubchild in subchild:
                    box[subsubchild.tag]=subsubchild.text
                    j=j+1
                if j==0 :
                    all_records[subchild.tag] = subchild.text
                else :
                    all_records[subchild.tag] = box
                i=i+1
            if(i==0):
                all_records[child.tag] = child.text
        else: return 0

    W = Image.open(base1+all_records["filename"])
    W = W.resize((200,200))
    W.save(base1+all_records["filename"])
    im = imageio.imread(base1+all_records["filename"])
    val=all_records["bndbox"]
    y=[]
    if all_records["name"]=='dog':
    
        temp=float(val["xmin"])
        temp=(temp/(float(all_records["width"]))) * 200
        temp=int(temp)
        y.append(temp)
        temp=float(val["xmax"])
        temp=(temp/(float(all_records["width"]))) * 200
        temp=int(temp)
        y.append(temp)
        temp=float(val["ymin"])
        temp=(temp/(float(all_records["height"]))) * 200
        temp=int(temp)
        y.append(temp)
        temp=float(val["ymax"])
        temp=(temp/(float(all_records["height"]))) * 200
        temp=int(temp)
        y.append(temp)
        #print y

        global count
        if count<200:
            X_train.append(im)
            Y_train.append(y)
        else:
            X_test.append(im)
            Y_test.append(y)
        count=count+1
    else :
        return 0

print "hi"   
k=0
for filename in os.listdir(base):
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
print Y_test.shape
model = Sequential()
model.add(Convolution2D(32, (5, 5), activation='relu', input_shape=(200,200,3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(4, activation='relu'))
model.compile(loss='mse',
              optimizer='adam',
              metrics=['accuracy',f1_score, precision, recall])

model.fit(X_test,Y_test, batch_size=32, epochs=32,verbose=1, validation_split=0.1)

print model.evaluate(X_train, Y_train, verbose=0)
def IOU(bbox1, bbox2):
    
    x1, x2, y1, y2 = bbox1[0], bbox1[1], bbox1[2], bbox1[3]
    x_1, x_2, y_1, y_2 = bbox2[0], bbox2[1], bbox2[2], bbox2[3]

    w_I = min(x2,x_2) - max(x1, x_1)
    h_I = min(y2, y_2) - max(y1, y_1)
    if w_I <= 0 or h_I <= 0:
        return 0
    I = w_I * h_I

    U = (x2-x1)*(y2-y1) 

    return I / U
y_pred=model.predict(X_train)
#confusion_matrix(Y_train, y_pred)
sum1=0
print y_pred[0], Y_train[0]
for i in range(len(X_train)):
    sum1=sum1+IOU(Y_train[i],y_pred[i])
print sum1/(len(X_train))

