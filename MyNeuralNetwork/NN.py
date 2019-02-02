import random
import math
import re

def sigm(x):
	return ( 1 / (1 + math.exp(-x)) )

def dSigm(x):
	return ( sigm(x) * (1-sigm(x)) )


class NeuralNetwork:
	
	def __init__(self, sturcture, learningRate):
		self.sturcture		= sturcture
		self.learningRate	= learningRate
		self.layers 		= len(sturcture)

		self.neurons = []		# containing neurons each layers
		self.costFunction = 0 	# containing the cost functino
		
		self.summCost = 0		# Summ of cost functions
		self.trainings = 0		# How many times was the network trained

		#create neurons and fill them layer, row and inputs informations
		self.createNeurons(self.sturcture, self.layers)
		pass

	def computeCostFunct(self, targets):
		# Cost of this training
		cost = 0

		# If the target is not matchin the output in number we raise an error!
		output_number = self.sturcture[self.layers-1]
		if(output_number != len(targets)):
			print("ERROR: Target number not equal with output neuron number!", len(targets), output_number)
		
		# Compute Cost values
		else:
			index = 0
			for outputNeuron in self.neurons[self.layers-1]:
				cost += math.pow( (outputNeuron.a - targets[index]), 2)
				index += 1

		return cost

	def train(self, inputs, targets):

		# Increase training counter
		self.trainings += 1

		# Fills neurons with output (.a) values
		currentAnswer = self.answer(inputs)

		# Summarize cost to get the cost for the whole training
		currentCost = self.computeCostFunct(targets)
		self.summCost += currentCost
		self.costFunction = self.summCost/self.trainings

		#print("Answer:", currentAnswer, " Target", targets, " currentCost", currentCost, " SummCost", self.costFunction)

		# Do the back propagation
		deltaCdict = self.backPropagation(targets)

		# Change the parameters arrocding to the result of the back propagation
		self.changeParameters(deltaCdict, currentCost)

		pass # Function

	def changeParameters(self, deltaCdict, costFunction):
		#Add the deltas to the parameters
		ilayer = 0
		for layer in self.neurons:
			if(ilayer == 0):
				# Skip the input layer there is nothing to change there
				pass
			else:
				j = 0
				for neuron in layer:
					k = 0
					weights = []

					for weight in neuron.getWeights():
						key = "w"+str(ilayer)+" "+str(j)+" "+str(k)
						weights.append(-1 * (deltaCdict[key] * self.learningRate * costFunction))	#cost function is the error
						k += 1

					neuron.changeWeights(weights)
					key = "b"+str(ilayer)+" "+str(j)
					neuron.changeBias(-1 * deltaCdict[key] * self.learningRate * costFunction)	#cost function is the error
					
					j += 1
				pass
			ilayer += 1

	def answer(self, inputs):
		for layer in range(len(self.neurons)):
			# Input layer
			if(layer == 0):
				for row in range(len(self.neurons[layer])):
					self.neurons[layer][row] = inputs[row]
			# First hidden layer
			elif(layer == 1):
				for neuron in self.neurons[layer]:
					neuron.computeOutput(self.neurons[0])
					pass
				pass
			# Rest of layers
			else:
				#get prev layer outputs
				previousLayerOutput = []
				for neuron in self.neurons[layer-1]:
					previousLayerOutput.append(neuron.a)
					pass
				for neuron in self.neurons[layer]:
					neuron.computeOutput(previousLayerOutput)
					pass
			pass
		
		retVal = []
		for outputNeuron in self.neurons[self.layers-1]:
			retVal.append(outputNeuron.getOutput())
		return retVal
	
	# Not scalable!
	def backPropagation(self, targets):

		deltaCdict = {}

		for layer in range(len(self.neurons)):
			L = (len(self.neurons) - 1) - layer

			# If we reached the output layer, escape we don't have any parameter there.
			if(L == 0):
				break;

			for j in range(len(self.neurons[L])):
				# Get delta a values
				key = "a"+str(L)+" "+str(j)
				if(L == (len(self.neurons) - 1)):	# Output layer
					deltaCdict[key] = 2 * (self.neurons[L][j].a - targets[j])

				else:	# Not the output layer
					deltaCdict[key] = 0
					# Go the previuos layer
					for i in range(len(self.neurons[L+1])):
						deltaCdict[key] += self.neurons[L+1][i].weights[j] * dSigm(self.neurons[L+1][i].zValue) * deltaCdict["a"+str(L+1)+" "+str(i)]
						pass
					pass

				# Get delta weights
				if(L == 1):	# Last hidden layer
					for k in range(len(self.neurons[L-1])):
						key = "w"+str(L)+" "+str(j)+" "+str(k)
						deltaCdict[key] = self.neurons[L-1][k] * dSigm(self.neurons[L][j].zValue) * deltaCdict["a"+str(L)+" "+str(j)]
						pass
					pass
				else:	# Not the last hidden layer
					for k in range(len(self.neurons[L-1])):
						key = "w"+str(L)+" "+str(j)+" "+str(k)
						deltaCdict[key] = self.neurons[L-1][k].a * dSigm(self.neurons[L][j].zValue) * deltaCdict["a"+str(L)+" "+str(j)]
						pass
					pass
				# Get delta bias
				key = "b"+str(L)+" "+str(j)
				deltaCdict[key] = dSigm(self.neurons[L][j].zValue) * deltaCdict["a"+str(L)+" "+str(j)]
			pass # For Layers
		
		return deltaCdict

		pass

	# Debug function: Shows the output of the neurons
	def showAnswer(self):
		print("\nNeuralNetwork::showAnswer")

		for layer in range(len(self.neurons)):
			print("layer:", layer)
			if(layer == 0):
				print("\tInputs:", self.neurons[layer], end='')
			elif(layer == 1):
				for neuron in self.neurons[layer]:
					print("\tNeuron.a", neuron.a, end='')
					pass
				pass
			else:
				for neuron in self.neurons[layer]:
					print("\tNeuron.a", neuron.a, end='')
					pass
			print()
			pass

	# To do
	def connectNetwork(self):

		pass

	# Create neurons
	def createNeurons(self, sturcture, layers):
		# self.neurons will caontain Layer number of arrays
		# The first is the input, the last is the output, middle is the hidden
		self.neurons = []
		# initialize self.neurons array
		for a in range(len(sturcture)):
			self.neurons.append([0 for b in range(sturcture[a])])
			pass

		#Go through layers, from 0 to Layer-1
		for layer in range(layers):
			# Leave the input layer out, as it is only numbers
			if(layer == 0):
				pass
			# Not the input layer
			if(layer != 0):
				for row in range(sturcture[layer]):
					#The input connections is the prev layer's node number
					input_connections = sturcture[layer-1]
					self.neurons[layer][row] = Neuron(
						layer=layer, 
						row=row, 
						input_connections=input_connections)
				pass #If
			pass #For loop
		pass #Function
		
	# Debug functions: Shows the parameters of the neurons
	def checkNeurons(self):
		# Test creation
		print("\nNeuralNetwork::CheckNeurons")
		ilayer = 0
		for layer in self.neurons:
			if(ilayer == 0):
				pass
			else:
				j = 0
				for neuron in layer:
					print(ilayer, j, " Weights:", neuron.getWeights())
					print(ilayer, j, " Bias:", neuron.getBias())
					print(ilayer, j, " Output:", neuron.getOutput())
					print()
					j += 1
					pass
				pass
			ilayer += 1
				
		pass

	# To do
	def visualiseNetwork(self):
		# Draw the network to visualize it
		pass

	# Debug function to see how a parameter change changes the network
	def changeParameter(self, answer_input, i_layer, i_row, i_weights, i_bias):
		self.answer(answer_input)
		self.checkNeurons()

		ilayer = 0
		for layer in self.neurons:
			j = 0
			for neuron in layer:
				if((ilayer == i_layer) and (j == i_row)):
					neuron.changeWeights(i_weights)
					neuron.changeBias(i_bias)
					pass
				j += 1
				pass
			ilayer += 1
			pass

		self.answer(answer_input)
		self.checkNeurons()

		pass
		

class Neuron:

	# Constructor
	def __init__(self, layer, row, input_connections):
		self.layer = layer							# In which layer is this neuron
		self.row = row								# In which row is this neuron
		self.input_connections = input_connections	# How many input connections does it has
		
		self.a = 0									# The output value of the Neuron
		self.weights = [0 for x in range(input_connections)]	# The weights of the neuron
		self.bias = 0								# The bias value
		
		self.zValue = 0								# Value of a*w+b

		# Set weight and bias randomly at init
		self.initWeights()
		self.initBias()

	def initWeights(self):
		# Set all weights to random number
		for i in range(len(self.weights)):
			self.weights[i] = (random.uniform(-1, 1))	#random number between -1, and 1

	def changeWeights(self, deltaWeights):
		# Here we are setting to weights accorind to the deltaC
		index = 0
		for weight in self.weights:
			self.weights[index] += deltaWeights[index]
			index += 1

	def getWeights(self):
		return self.weights
		pass

	def initBias(self):
		self.bias = random.uniform(-1, 1)
		pass

	def changeBias(self, deltaBias):
		# Set bias to specific value
		self.bias += deltaBias

	def getBias(self):
		return self.bias
		pass

	def computeOutput(self, inputs):
		# Compute the value of Z
		self.computeZ(inputs)
		# Get the sigmod of Z
		self.a = sigm(self.zValue)

	def computeZ(self, inputs):
		# Reset Z value
		self.zValue = 0

		# Check if len(inputs) == len(weights) and input_connections
		if (len(inputs) != len(self.weights)):
			print("ERROR: Neuron::output Different input values as weights!", len(inputs), len(self.weights))

		# Multiply inputs and Weights and summarise them
		for index in range(len(inputs)):
			self.zValue += self.weights[index] * inputs[index]

		# Add bias to Z too.
		self.zValue += self.bias

	def getOutput(self):
		return self.a
		pass

	pass