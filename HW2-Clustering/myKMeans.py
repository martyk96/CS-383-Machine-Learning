#!/usr/bin/env python3
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import copy
np.random.seed(100)

def standardize(X):
    """
    Usage:
        standardize(X)

    Parameters:
        X: data to be standardized
    
    Returns:
        the standardized data
    """ 
    stdX = X.sub(X.mean(), axis=1)
    stdX = stdX.div(X.std(), axis=1)
    return stdX

def d_eucl(p1,p2):    
    """
    Usage:
        d_eucl(p1,p2)

    Parameters:
        p1,p2: two points in cartesian space

    Returns:
        the euclidean distance between p1 and p2
    """
    squared_dist = np.sum((p1-p2)**2, axis=0)
    return np.sqrt(squared_dist)

def d_manh(p1,p2):   
    """
    Usage:
        d_manh(p1,p2)

    Parameters:
        p1,p2: two points in cartesian space

    Returns:
        the mahattan distance between p1 and p2
    """ 
    return np.sum(np.absolute(p1-p2))


def myKMeans(X,y,k):
    """
    Usage:
        myKMeans(X,y,k)

    Parameters:
        X: observable data, (numpy array)
        y: target labels (numpy array)
        k: number of clusters (int)

    Purpose:
        Creates an animation of the KMeans algorithm on the observable data X. Plots are created using matplotlib
    
    Returns: 
        void
    """ 
    COLOR = ['b','r','g','y','c','m','k']

    fig = plt.figure()
    
    _,c = X.shape
    if c > 3:
        pca = PCA(n_components=3)
        X = pca.fit_transform(X)
        ax = fig.add_subplot(111, projection='3d')
        dim = 3
    elif c == 3:
        ax = fig.add_subplot(111, projection='3d')
        dim = 3
    elif c == 2:
        ax = fig.add_subplot(111)
        dim = 2
    else:
        raise ValueError("X needs to have at least 2 dimensions")
    
    # randomly select k samples
    mean = X[np.random.randint(X.shape[0], size=k), :]
    
    labels = np.empty(len(y),dtype=object) 
       
    iteration = 0
    while iteration == 0 or sumManh > 2**-23:
        iteration += 1
    
        oldMean = copy.deepcopy(mean)
        for idx,row in enumerate(list(X)):
            minDist = math.inf
            for i in range(len(mean)):
                dist = d_eucl(row,mean[i])
                if dist < minDist:
                    minDist = dist
                    label = i
                labels[idx] = label
        # Update the reference vectors
        mean[label] = np.mean(X[np.where(labels==label)],axis=0)
        

        ax.cla()
        cluster_purity = []
        for label in set(labels):
            r = np.where(labels == label)
            if dim == 2:
                ax.scatter(X[r,0],X[r,1],c=COLOR[label],marker='x',alpha=0.3)
            else:
                ax.scatter(X[r,0],X[r,1],X[r,2],c=COLOR[label],marker='x',alpha=0.3)
            
            # clutser purity calc
            valCount = []
            for trueLabel in set(y):
                valCount.append(np.count_nonzero(y[r]==trueLabel))
            np.amax(valCount)
            cluster_purity.append(np.amax(valCount))
        purity = np.sum(cluster_purity)/len(y)

        # add mean points to plot
        if dim == 2:
            for i in range(len(mean)):
                ax.scatter(mean[i,0],mean[i,1],c='k',marker='o')    
        else:
            for i in range(len(mean)):
                ax.scatter(mean[i,0],mean[i,1],mean[i,2],c='k',marker='o')  

        ax.set_title('Iteration %d; Purity = %.5f' %(iteration,purity))

        # sum of the manhatten distances between last and current mean for each of the means
        sumManh = 0
        for m in range(len(mean)):
            sumManh += d_manh(oldMean[m],mean[m])

        plt.pause(0.5)
    # keep the plot open
    plt.show()

if __name__ == "__main__":
    headers = ['Class','TimesPregnant','PlasmaGlucoseConcentration','DiastolicBloodPressure','TricepsSkinFoldThinkness','Insulin','BMI','DiabetesPedigreeFunction','Age']
    df = pd.read_csv("diabetes.csv",names=headers)

    X = df.iloc[:,1:]
    y = df.Class

    stdX = standardize(X)

    myKMeans(np.array(stdX),np.array(y),2)
