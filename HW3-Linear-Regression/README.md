# HW 3 - Linear Regression
Martin Kraus - CS 383 Machine Learning

Checkout Assignment.pdf to see problem sets.

## Problem 1 - Theory
To run the code to create the 3d plot for this question simply run:
```bash
make problem1
```
    
## Problem 2 - Gradient Descent
To run the code to create the the iteration plots for J x1 and x2 simply run:
```bash
make problem2
```

## Problem 3 - Closed Form Linear Regression
To run the code associated with determining the model for a given dataset:
```bash
make problem3 datafile=[filepath]
```
Where `filepath` is the path to the dataset.
    
## Problem 4 - S-Folds Cross-Validation
To run the code associated with performing S-folds cross-validation, simply run:
```bash
make problem4 s=[S] datafile=[filepath]
```

Where `S` is the number of fold, and `filepath` is the path to the dataset.

## Datasets
Datasets must be in CSV format and has the first row be header information, the first column be some integer index, the second column as the target value, then D columns of real-valued features.

