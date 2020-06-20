from numpy import mean
from numpy import std
from matplotlib import pyplot
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD

#Loading the mnist dataset 
(trainX,trainy),(testX,testy)=mnist.load_data()

#Reshaping the matrix to have dimensions 60000,28,28,1, so there is only 1 color channel gray
trainX=trainX.reshape(trainX.shape[0],28,28,1)
testX=testX.reshape(testX.shape[0],28,28,1)

#There are 10 classes for each digit, this function will classify each class into binary ie first column will contain only on 1's indice
trainy=to_categorical(trainy)
testy=to_categorical(testy)

#Normalise the dataset to make it more scaleable
train_norm=trainX.astype('float32')
test_norm=testX.astype('float32')

trainX=train_norm/255.0
testX=test_norm/255.0

#Training the model
model=Sequential()
#Add a convolutional layer with filtersize of 28x28x1
model.add(Conv2D(32,(3,3),activation='relu',kernel_initializer='he_uniform',input_shape=(28,28,1)))
#Add a maxPooling layer
model.add(MaxPooling2D())
#Add a flatten Layer
model.add(Flatten())
#Add a dense layer of 100 nodes
model.add(Dense(100,activation='relu',kernel_initializer='he_uniform'))
#Final output must have 10 nodes so a dense layer of 10 nodes
model.add(Dense(10,activation='softmax'))
#Gradient Descent Optimization function with learning rate 0.01
opt=SGD(lr=0.01,momentum=0.9)
#Compile Model
model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])
#Fit the model 
model.fit(trainX,trainy,epochs=10,batch_size=32,verbose=0)
_, acc = model.evaluate(testX, testy, verbose=0)
print('> %.3f' % (acc * 100.0))
model.save('final_model.h5')
#Final model is ready with accuracy 98.05 , could be improved to 99.2%.