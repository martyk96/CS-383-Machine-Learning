# HW 2 - Clustering 
Martin Kraus - CS 383 Machine Learning

## myKMeans.py:
_myKMeans_ is a custom implementation of the KMeans clustering algorithm. 
_myKMeans_ takes two arguemnts, observable data **X** and labels **y**.

While KMeans is a _unsupervised_ algorithm, the scope of the assignment included the calcualtion of the purity requiring the need of the target labels

_Note:_ X and y must be numpy arrays. numpy module is included in the virtual environment

    
To test the function with my test parameters (k=2 for all features) (depends on venv below): 

```bash
make test
```

To just create the virtual environment and install requirements:
```bash
make venv
```
