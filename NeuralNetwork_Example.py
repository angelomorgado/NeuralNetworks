import numpy as np

#Sigmoid mathematical function will act as our activation function 
def sigmoid(x):
  return 1 / (1 + np.exp(-x))

#We will create a class to simulate a neuron
class Neuron:
  #The __init__ method is the equivalent of a constructor in c++
  def __init__(self, weights, bias):
    self.weights = weights
    self.bias = bias

  #This is the function that will receive the input and return the output of the neuron
  def feedforward(self, inputs):
    #Weight inputs, add bias and then use that value on the activation function
    #np.dot is the dot product of two arrays (x0*yo + x1*y1 + ... + xn*yn)
    total = np.dot(self.weights, inputs) + self.bias
    return sigmoid(total)


#We will create a class to simulate the neural network
class NeuralNetwork:
  '''
  A neural network with:
    - 2 inputs
    - a hidden layer with 2 neurons (h1, h2)
    - an output layer with 1 neuron (o1)
  Each neuron has the same weights and bias:
    - w = [0, 1]
    - b = 0
  '''

  def __init__(self):
    weights = np.array([0,1])
    bias = 0

    #Create the neurons that belong to the neural network
    self.h1 = Neuron(weights,bias)
    self.h2 = Neuron(weights,bias)
    self.o1 = Neuron(weights,bias)

  def feedforward(self, x):

    #Get the output of the hidden layer to use as input for the output layer
    out_h1 = self.h1.feedforward(x)
    out_h2 = self.h2.feedforward(x)

    #Get the output
    out_o1 = self.o1.feedforward(np.array([out_h1, out_h2]))
    return out_o1

    
#Our main function
if __name__ == '__main__':

  #Create our neural network
  network = NeuralNetwork()

  #Give input and printing the output using the feedforward
  x = np.array([2,3]) #x1 = 2, x2 = 3

  print(network.feedforward(x)) #should output 0.999 as predicted