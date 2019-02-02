from NN import Neuron
from NN import NeuralNetwork

def trainOR():
	Network0 = NeuralNetwork(sturcture=[2,3,1], learningRate=0.5)

	for i in range(1000):
		Network0.train([0,0], [0])
		Network0.train([0,1], [1])
		Network0.train([1,0], [1])
		Network0.train([1,1], [1])
		print(i,"",round((1-Network.costFunction)*100, 0))
		pass

	print(round((1-Network0.costFunction)*100, 2))

	Network0.answer([0,0])

	print(Network0.neurons[2][0].getOutput())
	pass

def trainXOR():
	print("\ntrain XOR")
	sturcture = [2,3,1]
	Network0 = NeuralNetwork(sturcture=sturcture, learningRate=1)

	for i in range(10000):
		Network0.train([0,0], [0])
		Network0.train([0,1], [1])
		Network0.train([1,0], [1])
		Network0.train([1,1], [0])

		#print(i,"",round((1-Network0.costFunction)*100, 0))
		print(i,"",round( abs(Network0.neurons[len(sturcture)-1][0].a - 0), 3))

	print( Network0.answer( [0,0] ) )
	print( Network0.answer( [1,0] ) )
	print( Network0.answer( [0,1] ) )
	print( Network0.answer( [1,1] ) )

	pass

def main():

	trainXOR()

	pass

if __name__ == "__main__":
	main()