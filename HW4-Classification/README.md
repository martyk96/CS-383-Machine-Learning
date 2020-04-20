# HW 4 - Classification
Martin Kraus - CS 383 Machine Learning

## Naive Bayes Classifier
The first coding part of this assigment asked for an implementation of a naive bayes classifier to train and classify a given dataset. 
The dataset has 57 continuous features followed by a class id of 0 or 1. 
The first 2/3 of the data are used to train while the remaining 1/3 of the data will be used for testing. 
The classifier will return the following statistics: precision, recall, f-measure, and accuracy.

To run:
```bash
make test_bayes datafile=[filepath]
```
Where filepath is the path to the dataset.

While a dataset was given, the code should work on any dataset that lacks header information and has several comma-separated continuous-valued features followed by a class id of 0 or 1.

## Logistic Regression
The second coding part of this assignment asked for an implementation of a logistic regression classifier to train and classify a given dataset
The dataset has 57 continuous features followed by a class id of 0 or 1. 
The first 2/3 of the data are used to train while the remaining 1/3 of the data will be used for testing. 
The classifier will return the following statistics: precision, recall, f-measure, and accuracy.

To run:
```bash
make test_logReg datafile=[filepath]
```
Where filepath is the path to the dataset.

While a dataset was given, the code should work on any dataset that lacks header information and has several comma-separated continuous-valued features followed by a class id of 0 or 1.