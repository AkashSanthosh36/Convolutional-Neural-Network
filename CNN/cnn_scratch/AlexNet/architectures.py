from process import Requirements
class Architectures:
	def AlexNet(x):
		conv_output1=Requirements.conv(x,filter_size=11,no_of_filters=96,stride=4,pad=0)
		pool_output1=Requirements.MaxPooling2D(conv_output1,pool_size=3)

		conv_output2=Requirements.conv(pool_output1,filter_size=5,no_of_filters=256,stride=1,pad=2)
		pool_output2=Requirements.MaxPooling2D(conv_output2,pool_size=3)

		conv_output3=Requirements.conv(pool_output2,filter_size=3,no_of_filters=384,stride=1,pad=1)
		conv_output4=Requirements.conv(conv_output3,filter_size=3,no_of_filters=384,stride=1,pad=1)
		conv_output5=Requirements.conv(conv_output4,filter_size=3,no_of_filters=256,stride=1,pad=1)

		pool_output3=Requirements.MaxPooling2D(conv_output5,pool_size=3)
		flattening=Requirements.flattening(pool_output3)
		print(flattening.shape)

		fc_output1=Requirements.forwardpropagation(flattening,hidden_layer_neurons=4096,output_neurons=4096)
		fc_output2=Requirements.forwardpropagation(fc_output1,hidden_layer_neurons=2000,output_neurons=1000)
		print(fc_output2)
		output=Requirements.softmax(fc_output2)

		return output
	






		"""conv_output1=Requirements.conv(x,filter_size=11,no_of_filters=96,stride=4,pad=0)
		activation_output1=Requirements.activation(pool_output1)

		conv_output2=Requirements.conv(activation_output1,filter_size=5,no_of_filters=16,stride=1,pad=0)
		pool_output2=Requirements.MaxPooling2D(conv_output2)
		activation_output2=Requirements.activation(pool_output2)

		flattening=Requirements.fully_connected(activation_output2)
		fc_output=Requirements.forwardpropagation(flattening,hidden_layer_neurons=120,output_neurons=10)
		output=Requirements.softmax(fc_output)
		return output"""
		return pool_output1
