import numpy as np
np.random.seed(0)

# suppress divide by zero for log warning, doesn't seem to impact calculation
import warnings
warnings.filterwarnings("ignore",message="divide by zero")
warnings.filterwarnings("ignore",message="invalid value")


class myLogReg():
    def __init__(self):
        self.theta = 0
        self.g = lambda x,th: 1/(1+np.exp(np.dot(-x,th)))

    def train(self,X,y,learningRate = 0.1):
        Jold = 0.1
        J = 0

        # transform y into a NDArray for correct math
        y = y.reshape(-1,1)

        # random thetas ranging from [-1,1]
        theta = np.random.random_sample((np.size(X,1),1))*2 - 1

        # run until J converges
        while abs(J-Jold) > 2**-32:
            Jold = J
            theta += learningRate/np.size(y,0) * np.dot(X.T,y-self.g(X,theta))

            # sum of the log liklihoods
            J = np.sum(y*np.log(self.g(X,theta))+(1-y)*np.log(1-self.g(X,theta)))
        
        # theta coverged, save to self
        self.theta = theta

    def predict(self,x):
        pred = self.g(x,self.theta).flatten()
        pred = np.where(pred<0.5,0,pred)
        pred = np.where(pred>=0.5,1,pred)
        return pred