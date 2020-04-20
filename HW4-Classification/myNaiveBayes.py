import numpy as np

class myNaiveBayes():
    def __init__(self):
        self.targets = ()
        self.targetProb = 0
        self.features = 0
        self.sig = []
        self.mu =[]

    def train(self,X,y):
        _,self.features = X.shape
        self.targets = np.array(list(set(y)))
        targetProb = []

        # split data by class
        for target in list(self.targets):
            # data associated w/ current target
            x = X[np.where(y == target)]

            # calculate the target probability (#of target/#of all)
            targetProb.append(np.size(x,0)/len(y))

            # calc the mu and sigma for each feature (will be used to generate the gaussian distribution) 
            self.mu.append(np.mean(x,axis=0))
            self.sig.append(np.sqrt(np.var(x,axis=0)))

        self.targetProb = np.array(targetProb)
    
    def _gaussian(self,x):
        calc_gauss = lambda x,i: 1/(self.sig[i]*np.sqrt(2*np.pi))*np.exp(-np.power(x-self.mu[i],2)/(2*np.power(self.sig[i],2)))
        for i in range(np.size(self.targets)):
            if i ==0:
                gauss = calc_gauss(x,i)
            else:
                gauss = np.vstack([gauss,calc_gauss(x,i)])
        return gauss

    def predict(self,x):
        predict = [] # array for the predicted value

        # for each test sample
        for row in x:
            # P(y=a) x P(f0=x[:,0]) x P(f1=x[:,1]) ... P(fD=x[:,D])
            prob = np.prod([self.targetProb,np.prod(self._gaussian(row.T),axis=1)],axis=0)

            # the target associated w/ the argmax(value) is the predicted class for this sample
            predict.append(self.targets[np.unravel_index(np.argmax(prob, axis=None), prob.shape)])
        
        return np.array(predict)
