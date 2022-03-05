import numpy as np

#Sigmoid mathematical function will act as our activation function 
def sigmoid(x):
  return 1 / (1 + np.exp(-x))

#The derivative of the sigmoid function (done in the paper)
def derivSigmoid(x):
  fx = sigmoid(x)
  return fx * (1 - fx)

#To calculate the loss we'll use the mean squared error
def mse_loss(yTrue, yPred):
  return ((yTrue - yPred) ** 2).mean()


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

  #The variables will start random so they can be tweaked and improve over time
  def __init__(self):
    #Weights
    self.w1 = np.random.normal()
    self.w2 = np.random.normal()
    self.w3 = np.random.normal()
    self.w4 = np.random.normal()
    self.w5 = np.random.normal()
    self.w6 = np.random.normal()

    #Biases
    self.b1 = np.random.normal()
    self.b2 = np.random.normal()
    self.b3 = np.random.normal()

  #Function that makes the path from the input to the output and returns it
  #x is a numpy array with 2 elements (weight abd height)
  def feedforward(self, x):

    #Get the output of the hidden layer to use as input for the output layer
    h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b1)
    h2 = sigmoid(self.w3 * x[0] + self.w4 * x[1] + self.b2)

    #Get the output
    o1 = sigmoid(self.w5 * h1 + self.w6 * h2 + self.b3)
    return o1

  #data is a (n*2) numpy array, n = number of samples in the dataset
  #all_y_trues is a numpy array with n elements containing the true values of the gender
  def train(self, data, all_y_trues):
    learnRate = 0.1 #controls how fast the network learns
    epochs = 1000 #number of times to loop through the entire dataset

    #Go through all the epochs
    for epoch in range(epochs):
      #the zip function combines the elements of two arrays in tuples, like zip(x,y) = ((x1,y1),...,(xn,yn))
      #basically it's looping through two lists at the same time
      for x, yTrue in zip(data, all_y_trues):
        #We'll need the values of the feedfoward function (not the best code but the simplest to understand)
        #Auxiliary values
        sumH1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
        h1 = sigmoid(sumH1)
        
        sumH2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
        h2 = sigmoid(sumH2)
        
        sumO1 = self.w5 * h1 + self.w6 * h2 + self.b3
        yPred = sigmoid(sumO1) #Because yPred = o1 = sigmoid(sumO1)
        
        #===============Calculate the partial derivatives===============
        # All formulas done on paper and then copied to here
        # dL_dW1 represents "partial L / partial w1"
        #These formulas were achieved using backpropagation using chain rule
        dL_dYPred = -2 * (yTrue - yPred) 

        #Neuron o1
        dYPred_dW5 = h1 * derivSigmoid(sumO1)
        dYPred_dW6 = h2 * derivSigmoid(sumO1)
        dYPred_dB3 = derivSigmoid(sumO1)

        dYPred_dH1 = self.w5 * derivSigmoid(sumO1)
        dYPred_dH2 = self.w6 * derivSigmoid(sumO1)

        #Neuron h1
        dH1_dW1 = x[0] * derivSigmoid(sumH1)
        dH1_dW2 = x[1] * derivSigmoid(sumH1)
        dH1_dB1 = derivSigmoid(sumH1)
        
        #Neuron h2
        dH2_dW3 = x[0] * derivSigmoid(sumH2)
        dH2_dW4 = x[1] * derivSigmoid(sumH2)
        dH2_dB2 = derivSigmoid(sumH2)

        #===============Update weights and biases===============
        # Gradient Descent w1 ← ( w1 - η * ∂L / ∂w1 )
        # η = learn rate
        # ∂L / ∂x is divided in the three formulas multiplied we see above for each variable using chain rule
        
        #Neuron h1
        self.w1 -= learnRate * dL_dYPred * dYPred_dH1 * dH1_dW1
        self.w2 -= learnRate * dL_dYPred * dYPred_dH1 * dH1_dW2
        self.b1 -= learnRate * dL_dYPred * dYPred_dH1 * dH1_dB1
        
        #Neuron h2
        self.w3 -= learnRate * dL_dYPred * dYPred_dH2 * dH2_dW3
        self.w4 -= learnRate * dL_dYPred * dYPred_dH2 * dH2_dW4
        self.b2 -= learnRate * dL_dYPred * dYPred_dH2 * dH2_dB2
        
        #Neuron o1
        self.w5 -= learnRate * dL_dYPred * dYPred_dW5
        self.w6 -= learnRate * dL_dYPred * dYPred_dW6
        self.b3 -= learnRate * dL_dYPred * dYPred_dB3

      #===============Calculate the total loss at the end of each epoch===============
      #Only enters when it is a multiple of 10 or 0
      if epoch % 10 == 0:
        #np.apply_along_axis applies the feedforward to all elements of the axis 1 of data
        #Basically we're trying to predict all values of our data to then see how's the loss
        yPreds = np.apply_along_axis(self.feedforward, 1, data)
        loss = mse_loss(all_y_trues, yPreds)
        print("Epoch %d loss: %.3f" % (epoch, loss))
        

#Our main function
if __name__ == '__main__':

  #Define dataset
  data = np.array([
    [-2,-1], # Alice
    [25,6], # Bob
    [17,4], # Charlie
    [-15,-6], #Diana
  ])

  all_y_trues = np.array([
    1, #Alice
    0, #Bob
    0, #Charlie
    1, #Diana
  ])

  #Train out neural network
  network = NeuralNetwork()
  network.train(data, all_y_trues)

  #Predictions
  emily = np.array([-7,-3]) # 128 pounds, 63 inches
  frank = np.array([20,2]) # 155 pounds, 68 inches

  print("Emily: %.3f" % network.feedforward(emily))
  print("Frank: %.3f" % network.feedforward(frank))