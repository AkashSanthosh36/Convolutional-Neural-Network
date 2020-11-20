#importing required packages
import tensorflow as tf
import numpy as np
from keras.utils import np_utils
#reading and preparing the datasets
mnist=tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train = np.array(x_train)
x_test = np.array(x_test)

#Reshape the training and test set
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

#creating the convlolution neural network
class LeNet5:
	def __init__(self,x,y):
		self.x=x
		self.y=y
		self.model=tf.keras.models.Sequential()
	def call(self):
		#creating the lenet5

		#Layer1
		self.model.add(tf.keras.layers.Conv2D(filters=6,kernel_size=(5,5),strides=1,
					   padding="valid"))
		self.model.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))
		self.model.add(tf.keras.layers.ReLU())

		#Layer2
		self.model.add(tf.keras.layers.Conv2D(filters=16,kernel_size=(5,5),strides=1,
					   padding="valid"))
		self.model.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))
		self.model.add(tf.keras.layers.ReLU())

		#flattening
		self.model.add(tf.keras.layers.Flatten())

		#neural network
		self.model.add(tf.keras.layers.Dense(units=120,activation="relu"))
		self.model.add(tf.keras.layers.Dense(units=10,activation='softmax'))
		
		#optimizer
		self.model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
		#training
		self.model.fit(self.x,self.y,batch_size=100,steps_per_epoch=10,epochs = 2)	
emp=LeNet5(x=x_train,y=y_train).call()


