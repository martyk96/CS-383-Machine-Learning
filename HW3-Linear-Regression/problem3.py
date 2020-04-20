#!/usr/bin/env python3
import sys
import pandas as pd
import numpy as np
from closedLinReg import closedLinReg

# get the file from cmd line
datafile = sys.argv[1]

# rmse calculation
rmse = lambda Y,Yhat: np.sqrt((1/len(Y))*np.sum(np.square(Y-Yhat),axis=None))

# get the data
df = pd.read_csv(datafile)

# randomize
df = df.sample(frac=1,random_state=0)

# create test and trian sets
idx = int(2*len(df.Index)/3)

# extract features and target
X = df.iloc[:,2:]
y = df.iloc[:,1]

# split X into training and testing sets
train_X = np.column_stack((np.ones(idx),X.iloc[0:idx,:]))
train_y = np.array(y[0:idx])
test_X = np.column_stack((np.ones(len(df.Index)-idx),X.iloc[idx:,:]))

# create the model to train and predict
model = closedLinReg()
model.train(train_X,train_y)
pred = model.predict(test_X)

test_y = np.array(y[idx:])

print("RMSE: %.3f" % rmse(test_y,pred))
print("Theta: ", end="")
print(model.theta)
