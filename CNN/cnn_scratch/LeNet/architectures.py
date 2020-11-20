import numpy as np
from process import Requirements
import matplotlib.pyplot as plt
class Architectures:
	def LeNet(x,y,lr=0.1):
		filter1,filter2,weight=None,None,None
		for i in range(1):
			conv_output1,filter1=Requirements.conv(x,filter_size=5,no_of_filters=6,stride=1,pad=0,filter=filter1)
			pool_output1=Requirements.MaxPooling2D(conv_output1)
			activation_output1=Requirements.activation(pool_output1)
		
			conv_output2,filter2=Requirements.conv(activation_output1,filter_size=5,no_of_filters=16,stride=1,pad=0,filter=filter2)
			pool_output2=Requirements.MaxPooling2D(conv_output2)
			activation_output2=Requirements.activation(pool_output2)

			flattening=Requirements.flattening(activation_output2)
			if(weight==None):
				weight0,weight1,bias0,bias1=Requirements.weights_initialising(flattening,hidden_layer_neurons=120,output_neurons=10)
			fc_output,hidden_layer_output=Requirements.forwardpropagation(flattening,weight0,weight1,bias0,bias1)
			output=Requirements.softmax(fc_output)
			#backpropagation
			
			#softmax backpropagation
			error=output-y
			#neural backprop
			d_hiddenlayer,d_output=Requirements.backpropagation(fc_output,hidden_layer_output,weight1,error)

			pool2=Requirements.MaxPooling2D_backprop(conv_output2,pool_output2)
			convolve_backprop2=Requirements.convolve_backprop(pool2,activation_output1,filter2)
			print(convolve_backprop2.shape)
			
			"""pool1=Requirements.MaxPooling2D_backprop(conv_output1,pool_output1)
			convolve_backprop1=Requirements.convolve_backprop()

			#filter weight updation
			for f in range(filter1.shape[0]):
				filter1[f]=filter1[f]-lr*convolve_backprop1[f]
			for f in range(filter2.shape[0]):
				filter2[f]=filter2[f]-lr*convolve_backprop2[f]		
			#neural network weight updation
			print(filter2[0])
			weight0=weight0+(np.dot(flattening,d_hiddenlayer.T)*lr)
			weight1=weight1+(np.dot(hidden_layer_output,d_output.T)*lr)
			weight=1	"""
		return output