import numpy as np

#This error function was made to get the loss of a neural network
#The .mean() method uses all the values in the vector to calculate the mean
#The (y_true - y_pred) creates a new vector with  the values of y_true[n] - y_pred[n] 
def meanSquaredError(y_true, y_pred):
  return ((y_true - y_pred) ** 2).mean()

if __name__ == '__main__':
  y_true = np.array([1,0,0,1])
  y_pred = np.zeros(4) # or np.array([0,0,0,0])
  print(meanSquaredError(y_true, y_pred))