#!/usr/bin/env python3
import sys
import pandas as pd
import numpy as np
from closedLinReg import closedLinReg

def sqErrSFolds(X,y,S):
    model = closedLinReg()
    sqErr=[]
    
    # numpy to split the data into S-folds
    X = np.array_split(X,S)
    y = np.array_split(y,S)

    # for each fold, extract testing and training data
    for s in range(S):
        train_X = np.copy(X)
        train_y = np.copy(y)

        # split into train/test
        train_X = np.vstack(np.delete(train_X,s,0))
        train_y = np.hstack(np.delete(train_y,s,0))

        test_X = X[s]
        test_y = y[s]

        # train the model
        model.train(train_X,train_y)

        # use the model for the prediction
        pred = model.predict(test_X)

        # calculate the square error
        sqErr.append(np.square(pred-test_y))

    # return the rmse
    return np.sqrt(np.sum(np.hstack(sqErr))/len(np.hstack(sqErr)))

if __name__ == "__main__":
    # S from cmdline
    S = int(sys.argv[1])

    # datafile from cmd line
    datafile = sys.argv[2]

    df = pd.read_csv(datafile)
    N = len(df.Index)
    randState = 0

    rmse = []
    # for 20 iterations
    for i in range(20):
        # randomize
        df = df.sample(frac=1,random_state=randState)
        randState+=1

        # extract features and target
        X = np.array(df.iloc[:,2:])
        y = np.array(df.iloc[:,1])

        rmse.append(sqErrSFolds(X,y,S))
        
    print("RMSE avg: %.4f" % np.mean(rmse))
    print("RMSE std: %.4f" % np.std(rmse))