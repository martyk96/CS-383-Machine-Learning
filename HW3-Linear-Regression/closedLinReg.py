import numpy as np
class closedLinReg():
    def __init__(self):
        self.theta = []
        
    def train(self,train_X,train_y):
        self.theta = np.dot(np.linalg.inv(np.dot(np.transpose(train_X),train_X)),np.dot(np.transpose(train_X),train_y))
        
    def predict(self,test_X):
        return np.dot(test_X,self.theta)