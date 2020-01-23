import pandas as pd
import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt
import os
import shutil
from PIL import Image
import cv2
from keras.layers import Dense,GlobalAveragePooling2D
from keras.applications import MobileNet, resnet50
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelBinarizer
from keras.applications.mobilenet import decode_predictions
from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Lambda
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.optimizers import Adam
import gzip, pickle
from tqdm import tqdm


#get data: here we will use the tha data; byclass

with gzip.open('../emnist-gzip/gzip/emnist-byclass-train-images-idx3-ubyte.gz', 'r') as f:
    train_data=f.read(697940*784)#pickle.load(f, encoding='latin1')
#as we can see, the data is a .idx3-ubyte type after unzipping
train_data= np.frombuffer(train_data, dtype=np.uint8).astype(np.float32)

with gzip.open('../emnist-gzip/gzip/emnist-byclass-train-labels-idx1-ubyte.gz', 'r') as f:
    train_labels=f.read()

train_labels= np.frombuffer(train_labels, dtype=np.uint8).astype(np.float32)
train_data=train_data[16:].reshape(697932,28*28) #the data has an offset of 16 and 8 for the labels
train_labels=train_labels[8:]

#Save a sample
n=6
(fig, ax) = plt.subplots(8, n, figsize=(16, 10))
titles=[f'This is {m}' for m in train_labels[:8*n]]

for (i, l) in enumerate(train_data[:8*n]):
    # plot the loss for both the training and validation data
    ax[i//n,i%n].set_title(f"{titles[i]}")
    g=l.reshape((28,28))
    ax[i//n,i%n].imshow(g,cmap='Greys')
    ax[i//n,i%n].set_xticks([])
    ax[i//n,i%n].set_yticks([])

plt.figure(figsize=(12,20))
fig.savefig("images/samples.png")
plt.show()




#Or we could do prepocessing:
#space between two side or content quantity to be able to distinguis say 2, z and Z which
#will pretty much look the same
#preprocessing and transfer learn take enormous time
#We will go with vanilla


Batch_Size=150
Epochs=10
Validation_split=0.2
numCategories=62
chanDim=-1
inputs=Input(shape=(28,28))
x = Dense(288)(inputs)

x = Conv1D(320, (3), padding="same")(x)
x = Activation("relu")(x)
x = BatchNormalization(axis=chanDim)(x)
x = MaxPooling1D(pool_size=(3))(x)
x = Dropout(0.25)(x)

x = Conv1D(64, (2), padding="same")(x)
x = Activation("relu")(x)
x = BatchNormalization(axis=chanDim)(x)
x = Conv1D(64, (3), padding="same")(x)
x = Activation("relu")(x)
x = BatchNormalization(axis=chanDim)(x)
x = MaxPooling1D(pool_size=(2))(x)
x = Dropout(0.25)(x)

x = Conv1D(128, (3), padding="same")(x)
x = Activation("relu")(x)
x = BatchNormalization(axis=chanDim)(x)
x = Conv1D(128, (3), padding="same")(x)
x = Activation("relu")(x)
x = BatchNormalization(axis=chanDim)(x)
x = MaxPooling1D(pool_size=(2))(x)
x = Dropout(0.25)(x)
x = Flatten()(x)
x = Dense(256)(x)
x = Activation("relu")(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(numCategories)(x)
output = Activation('softmax', name="id_output")(x)

model=Model(inputs,output)
model.compile(
loss='sparse_categorical_crossentropy',
optimizer=Adam(lr=0.33,epsilon=1e-8, decay=0.995),
metrics=['accuracy'])
r=model.fit(train_dataM,train_labels, batch_size=Batch_Size,epochs=Epochs, validation_split=Validation_split, verbose=1)

folder='images'
Names = ["acc", "loss"]
plt.style.use("ggplot")
(fig, ax) = plt.subplots(1, 2, figsize=(18, 8))
titles=['Accuracies', 'Losses']
# loop over the accuracy names
for (i, l) in enumerate(Names):
    # plot the loss for both the training and validation data
    ax[i].set_title(titles[i])
    ax[i].set_xlabel("Epoch #")
    ax[i].set_ylabel(titles[i])
    ax[i].plot(np.arange(0, Epochs), r.history[l], label='train_'+l)
    ax[i].plot(np.arange(0, Epochs), r.history["val_" + l],
        label="val_" + l)
    ax[i].legend()

plt.figure(figsize=(12,20))
plt.show()
# save the accuracies figure
plt.tight_layout()
fig.savefig(f"{folder}/accs.png")
plt.close()

#Testing the model
with gzip.open('C:/Users/My PC/Documents/Techfield/emnist-gzip/gzip/emnist-byclass-test-images-idx3-ubyte.gz', 'r') as f:
    test_data=f.read(697940*784)

test_data= np.frombuffer(test_data, dtype=np.uint8).astype(np.float32)

with gzip.open('C:/Users/My PC/Documents/Techfield/emnist-gzip/gzip/emnist-byclass-test-labels-idx1-ubyte.gz', 'r') as f:
    test_labels=f.read()

test_labels= np.frombuffer(test_labels, dtype=np.uint8).astype(np.float32)

test_data=test_data[16:].reshape(116323,28,28)
test_labels=test_labels[8:]

result=model.predict(test_data)
result1=[result[i].argmax() for i in range(116323)]
Accuracy=np.mean([result1[i]==test_labels[i] for i in range(116323)])


def one_hot_encode(y):
    N = len(y)
    K = 62

    Y = np.zeros((N,K))

    for i in range(N):
        Y[i,int(y[i])] = 1

    return Y

#confusion matrix
conf=one_hot_encode(result1).T.dot(one_hot_encode(test_labels))
conf=pd.DataFrame(conf)
cc=np.zeros(62)
cl=np.zeros(62)
for i in range(62):
    cc[i]=(conf.loc[i,:].sum()-conf.loc[i,i])/conf.loc[i,:].sum()
    cl[i]=conf.iloc[i,:62].sum()
conf['%error']=cc*100
conf['total']=cl

conf.to_csv('C:/Users/My PC/Documents/GitHub/EMNIST analysis/confusion.csv')
