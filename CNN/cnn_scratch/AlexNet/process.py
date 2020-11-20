import numpy as np
import math
class Requirements:
	def conv(x,filter_size,no_of_filters,stride=1,pad=0):
		np.random.seed(0)
		#Padding
		new_image_size=x.shape[1]+2*pad
		img=np.zeros(shape=(3,new_image_size,new_image_size))
		for i in range(3):
			img[i,pad:new_image_size-pad,pad:new_image_size-pad]=x[i]
		
		#output
		output_size=int( (new_image_size-filter_size)/stride+1 )
		output=np.zeros(shape=(no_of_filters,output_size,output_size))

		#filter
		filter=np.random.uniform(low=0,high=1,size=(no_of_filters,3,filter_size,filter_size))
		
		#convolving
		for f in range(no_of_filters):
			r=0
			for i in range(0,output_size*stride,stride):
				c=0
				for j in range(0,output_size*stride,stride):
					curr_region=img[:,i:filter_size+i,j:filter_size+j]	
					output[f,r,c]=np.sum(curr_region*filter[f,:,:,:])
					c+=1
				r+=1	
		return output

	def MaxPooling2D(img,pool_size=2,stride=2):
		output_size=int( (img.shape[1]-pool_size)/stride+1 )
		output=np.zeros(shape=(img.shape[0],output_size,output_size))
		for f in range(0,img.shape[0],1):
			r=0
			for i in range(0,output_size*stride,stride):
				c=0
				for j in range(0,output_size*stride,stride):
					output[f,r,c]=np.max( img[f,i:pool_size+i,j:pool_size+j] )	
					c+=1
				r+=1							
		return output

	def activation(img):
		for f in range(img.shape[0]):
			for i in range(img.shape[1]):
				for j in range(img.shape[2]):
					if(img[f,i,j]<0):
						img[f,i,j]=0
		return img						
		
	def flattening(img):
		output_size=img.shape[0]*img.shape[1]*img.shape[2]
		output=np.zeros(shape=(output_size,1))
		d=0
		for f in range(img.shape[0]):
			for i in range(img.shape[1]):
				for j in range(img.shape[2]):
					output[d]=img[f,i,j]		
					d=d+1
		return output

	def tanh(img):
		return np.tanh(img)

	def relu(img):
		for i in range(img.shape[0]):
			for j in range(img.shape[1]):
				if(img[i,j]<0):
						img[i,j]=0
		return img				
			

	def forwardpropagation(img,hidden_layer_neurons=5,output_neurons=1):
		weight0=np.random.uniform(low=0,high=1,size=(img.shape[0],hidden_layer_neurons))
		weight1=np.random.uniform(low=0,high=1,size=(hidden_layer_neurons,output_neurons))
		bias0=np.random.uniform(low=0,high=1,size=(hidden_layer_neurons,1))
		bias1=np.random.uniform(low=0,high=1,size=(output_neurons,1))
		h0=np.dot(weight0.T,img)+bias0
		h1=Requirements.tanh(h0)
		w0=np.dot(weight1.T,h1)+bias1
		w1=Requirements.tanh(w0)
		return w1

	def softmax(img):
		return np.exp(img)/(np.sum(np.exp(img)))	

    	


