#importing required packages
import tensorflow as tf
import keras
tf.enable_eager_execution()

x=tf.truncated_normal([1,32,32,1])
y=tf.nn.softmax((tf.truncated_normal([1,1])))
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
		self.model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
		self.model.fit(self.x ,self.y, steps_per_epoch = 10, epochs = 42)

emp=LeNet5(x,y).call()

