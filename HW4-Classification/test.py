#!/usr/bin/env python3
import sys
import pandas as pd
import numpy as np
from myNaiveBayes import myNaiveBayes
from myLogReg import myLogReg

def standardize(X):    
    stdX = X.sub(X.mean(), axis=1)
    stdX = stdX.div(X.std(), axis=1)
    return stdX

# get the classification from cmd line
classifier= sys.argv[1]
# classifier = "bayes"

# get the file from cmd line
datafile = sys.argv[2]
# datafile = "spambase.data"

# get the data
df = pd.read_csv(datafile)

# randomize
df = df.sample(frac=1,random_state=0)

# create test and trian sets
idx = int(2*len(df.index)/3)

# extract features and target
X = standardize(df.iloc[:,1:-1])
y = df.iloc[:,-1]

# extact training/testing data from X and classify based on specified classifier
if classifier == "bayes":
    # split X into training and testing sets (no bias)
    train_X = np.array(X.iloc[0:idx,:])
    test_X = np.array(X.iloc[idx:,:])
    train_y = np.array(y[0:idx])
    test_y = np.array(y[idx:])      


    # create the model, train, and predict
    bayes = myNaiveBayes()
    bayes.train(train_X,train_y)
    prediction = bayes.predict(test_X)
elif classifier == "logReg":
    # split X,y into training and testing sets (add bias)
    train_X = np.column_stack((np.ones(idx),X.iloc[0:idx,:]))
    test_X = np.column_stack((np.ones(len(df.index)-idx),X.iloc[idx:,:]))
    train_y = np.array(y[0:idx])
    test_y = np.array(y[idx:])

    # create the model, train, and predict
    logreg = myLogReg()
    logreg.train(train_X,train_y)
    prediction = logreg.predict(test_X)
else:
    print("Invalid classifier, select either 'bayes' or 'logReg'")
    sys.exit(1)

# prediction statisctics
A = prediction
B = test_y

TP = np.count_nonzero(np.array(A == 1) & np.array(A==B))
TN = np.count_nonzero(np.array(A == 0) & np.array(A==B))
FP = np.count_nonzero(np.array(A == 1) & np.array(A!=B))
FN = np.count_nonzero(np.array(A == 0) & np.array(A!=B))

precision = TP/(TP+FP)
recall = TP/(TP+FN)
fmeasure = 2*(precision*recall)/(precision+recall)
accuracy = np.count_nonzero(prediction == test_y)/np.size(test_y)

print("Precision: %.5f" %(precision))
print("   Recall: %.5f" %(recall))
print("F-Measure: %.5f" %(fmeasure))
print(" Accuracy: %.5f" %(accuracy))

