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


#Our main function
if __name__ == '__main__':

  #Create our neuron
  weights = np.array([0,1]) #w1 = 0 , w2 = 1
  bias = 4
  neuron = Neuron(weights, bias)

  #Give input and printing the output using the feedforward
  x = np.array([2,3]) #x1 = 2, x2 = 3

  print(neuron.feedforward(x)) #should output 0.999 as predicted