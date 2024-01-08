import pandas as pd
import csv
import numpy as np
import sklearn
import imblearn

from numpy import loadtxt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import *
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.mixture import BayesianGaussianMixture
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import SGDClassifier
from sklearn.mixture import BayesianGaussianMixture
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LinearRegression
from collections import Counter
from imblearn.over_sampling import BorderlineSMOTE
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier

#Location of the input dataset
fileLocation ='food-drug-dataset.csv'

#Converting the dataset from CSV to a numpy array for passing it as input to learning models
def readDataFromFile(fileLocation):
  interactionData = pd.read_csv(fileLocation,
      names=['Food-id','Drug-id','Tanimoto','Dice','Cosine','Sokal','Tversky','Result'])
  interactionArray = np.array(interactionData)
  print(interactionArray[0])
  return interactionArray

#calling the function to get data
interactionArray = readDataFromFile(fileLocation)

#this function calculates the F1 score when test data input (y_test) and test data output(y_pred) values are given
def calculateF1Score(y_test, y_pred):
  weighted = f1_score(y_test, y_pred, average='weighted')
  return weighted

#this function splits the dataset into training and testing data to feed the learning algorithm
def prepareDataset(interactionArray): 
  X = interactionArray[:,2:7]
  y = interactionArray[:,7]
  y = y.astype(int)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
  return X_train, X_test, y_train, y_test


""" Prepare a dataset with k-folds. Currently taking k = 4.
The dataset is divided into k number of groups, each group has an equal number of samples. 
The 'k-1' groups are used for the model to train and the other one-fold is used for testing the model. 
Parameters: an array consisting of all the interactions and their coefficients
Returns: X_train, X_train, y_train, y_test are the array of training and testing datasets. Counter is number of k-splits."""
def prepareDatasetKFold(interactionArray):
  # Copy interactionArray into other array
  array = np.copy(interactionArray)
  # Splitting the array and taking only coeffecients, Tanimoto, Dice, Cosine, Sokal, Tversky indexes
  X = array[:,2:7]
  # Splitting the array and taking the labels as 'TRUE' or 'FALSE' values
  y = interactionArray[:,7]
  # Converting the 'TRUE' or 'FALSE' values into binary labels.
  y = y.astype(int)
  
  # Applying K-Fold cross validation
  kf = KFold(n_splits=4)
  kf.get_n_splits(X)

  #Initializing X_train, X_test, y_train, y_test as empty arrays 
  X_train, X_test, y_train, y_test = [], [], [], []

  # For each train and test index in split of X, y : Append X[train_index], y[train_index] to train dataset and X[test_index], y[test_index] to test dataset
  for train_index, test_index in kf.split(X, y):
    # print("TRAIN:", train_index, "TEST:", test_index)
    X_train.append(X[train_index])
    X_test.append(X[test_index])
    y_train.append(y[train_index])
    y_test.append(y[test_index])
  # Return the test and train dataset with k-fold (currently 4)
  return X_train, X_test, y_train, y_test, 4
  

""" This is an unsupervised clustering algorithm. It takes the number of clusters as parameters and clusters the data based on it.
The k-Means algorithm focuses on minimizing the inertia between clusters to choose a centroid.
Parameters: an array consisting of all the interactions and their coefficients
"""
def kMeansMethod(interactionArray):
  print('\n****** K-Means Method ******\n')
  # Get X_train, X_test, y_train, y_test variables with train-test-split method
  X_train, X_test, y_train, y_test = prepareDataset(interactionArray)
  # Initialize the KMeans algorithm implemented in sklearn
  model = KMeans(n_clusters=2)
  # Train the model with the training data
  model.fit(X_train)
  # Predict a response to the testing data
  y_pred=model.predict(X_test) 
  # Calculate the metrics using functions in sklearn
  print('Accuracy Score : ', metrics.accuracy_score(y_test,y_pred))
  print('F1 Score : ', calculateF1Score(y_test, y_pred))
  
kMeansMethod(interactionArray)

"""
The Gaussian algorithm is based on probabilities, it is also an unsupervised learning algorithm. 
This model implements an expectation-maximization algorithm, where we can use four ways to calculate 
the covariance which are, spherical, diagonal, tied, and full.
Parameters: an array consisting of all the interactions and their coefficients
"""
def gaussian(interactionArray):
  print('\n****** Gaussian Method ******\n')
  # Get X_train, X_test, y_train, y_test variables with train-test-split method
  X_train, X_test, y_train, y_test = prepareDataset(interactionArray)
  # Initialize the Gaussian algorithm implemented in sklearn
  model = GaussianMixture(n_components=2)
  # Train the model with the training data
  model.fit(X_train)
  # Predict a response to the testing data
  y_pred=model.predict(X_test) 
  # Calculate the metrics using functions in sklearn
  print('Accuracy Score : ', metrics.accuracy_score(y_test,y_pred))
  print('F1 Score : ', metrics.f1_score(y_test,y_pred))

gaussian(interactionArray)


"""The Bayesian Gaussian algorithm is a variation of the gaussian algorithm when inference is introduced. 
Inference helps the learning model by maximizing the lower bounds of the model which decreases the data likelihood. 
 Parameters: an array consisting of all the interactions and their coefficients.
"""
def bayesianGaussianMethod(interactionArray):
  print('\n****** Bayesian Gaussian Method ******\n')
  # Get X_train, X_test, y_train, y_test variables with train-test-split method
  X_train, X_test, y_train, y_test = prepareDataset(interactionArray)
  # Initialize the Gaussian algorithm implemented in sklearn
  model = BayesianGaussianMixture(n_components=2)
  # Train the model with the training data
  model.fit(X_train)
  # Predict a response to the testing data
  y_pred=model.predict(X_test) 
  # Calculate the metrics using functions in sklearn
  print('Accuracy Score : ', metrics.accuracy_score(y_test,y_pred))
  print('F1 Score : ', calculateF1Score(y_test, y_pred))

bayesianGaussianMethod(interactionArray)

"""This algorithm is a variant of the k-means algorithm that uses mini-batches of data to optimize computation. 
This method reduces the computation time taken to reach a local solution.
Parameters: an array consisting of all the interactions and their coefficients """
def minibatchKMeans(interactionArray):
  print('\n ****** Mini-Batch K-Means ******\n')
  #  X_train, X_test, y_train, y_test are variables havind training data and testing data
  X_train, X_test, y_train, y_test = prepareDataset(interactionArray)

  # Initializing MiniBatch K-means with 5 clusters, and batch size 6
  kmeans = MiniBatchKMeans(n_clusters=5, random_state=0, batch_size=6, n_init=3)
  # Train the model using fit() with X_train as a parameter
  kmeans = kmeans.fit(X_train, y_train)

  # Predict the response
  y_pred = kmeans.predict(X_test)
  # Calculate metrics using the metrics functions in sklearn
  print('Accuracy Score : ', metrics.accuracy_score(y_test, y_pred))
  print('F1 Score : ', metrics.f1_score(y_test, y_pred, average='weighted'))

minibatchKMeans(interactionArray)

""" Multi-layer Perceptron is one of the many neural network algorithms which contains multiple layers in the model. 
The first layer is the input layer, followed by a hidden layer of non-linear neuron layers.
Parameters: an array consisting of all the interactions and their coefficients. """
def multiLayerPerceptron(interactionArray):
  print('\n****** Multi Layer Perceptron ******\n')
  X_train, X_test, y_train, y_test = prepareDataset(interactionArray)
  print('Original dataset shape %s' % Counter(y_train))
  sm = BorderlineSMOTE(random_state=42)
  X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
  print('Resampled dataset shape %s' % Counter(y_train_res))
  model = MLPClassifier(hidden_layer_sizes=(8,8,8),activation="relu" ,random_state=1, max_iter=500).fit(X_train_res, y_train_res)
  y_pred=model.predict(X_test)
  print('Accuracy Score : ', metrics.accuracy_score(y_test,y_pred))
  print('F1 Score : ', metrics.f1_score(y_test,y_pred))

multiLayerPerceptron(interactionArray)

""" This algorithm is a supervised learning algorithm, it implements the gaussian method for classification. 
It depends on probabilistic functions and the major functionality of this function is that it assumes that 
features are independent of each other. 
Parameters: an array consisting of all the interactions and their coefficients
"""
def gaussianNaivesBayes(interactionArray):
  print('\n ****** Gaussian Naive Bayes ****** \n')

  # X_train, X_test, y_train, y_test are arrays having the k-fold datasets. Counter is the number of k that are used
  X_train, X_test, y_train, y_test, counter = prepareDatasetKFold(interactionArray)

  # Initialize precision, accuracy and f1-score as empty arrays
  precision, accuracy, f1_score = [], [], []
  
  # For each k in k-fold, fit the model, predict, and calculate metrics
  for i in range(counter):
    # Original dataset shape
    # print('Original dataset shape %s' % Counter(y_train[i]))

    # Using BorderlineSMOTE for upsampling the dataset
    sm = BorderlineSMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train[i], y_train[i])

    # Resampled dataset shape
    # print('Resampled dataset shape %s' % Counter(y_train_res))

    # Train the model using fit() with X_train_res and y_train_res and predict a response
    gnb = GaussianNB()
    y_pred = gnb.fit(X_train_res, y_train_res).predict(X_test[i])

    #Calculate the metrics using functions in sklearn
    precision.append(metrics.precision_score(y_test[i], y_pred))
    accuracy.append(metrics.accuracy_score(y_test[i], y_pred))
    f1_score.append(metrics.f1_score(y_test[i], y_pred))
  
  # Print accuracy, precision and f1-scores by taking the mean of the respective arrays
  print('Accuracy Score : ', np.mean(accuracy))
  print('Precision Score : ', np.amax(precision))
  
gaussianNaivesBayes(interactionArray)

""" This algorithm is a simple yet efficient method of supervised learning. When the data is given 
to this model it transforms the data into 10^5 training samples.
Parameters: an array consisting of all the interactions and their coefficients """
def stochasticGradientDescent(interactionArray):
  print('\n ****** Stochastic Gradient Descent ******\n')
  #  X_train, X_test, y_train, y_test are variables havind training data and testing data
  X_train, X_test, y_train, y_test, counter = prepareDatasetKFold(interactionArray)
  accuracy, f1_score = [], []

  for i in range(counter):
    # print('Original dataset shape %s' % Counter(y_train[i]))
    sm = BorderlineSMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train[i], y_train[i])
    # print('Resampled dataset shape %s' % Counter(y_train_res))

    clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=100000)
    clf.fit(X_train_res, y_train_res)
    y_pred=clf.predict(X_test[i])
    accuracy.append(metrics.accuracy_score(y_test[i], y_pred))
    f1_score.append(metrics.f1_score(y_test[i], y_pred))

  print('Accuracy Score : ', np.mean(accuracy))
  print('F1 Score : ', np.mean(f1_score))

stochasticGradientDescent(interactionArray)

""" The Linear Regression algorithm is a linear approach to model relationships between data points. 
It is based on conditional probability distribution where models depend linearly on unknown parameters or features.
Parameters: an array consisting of all the interactions and their coefficients """
def linearRegression(interactionArray):
  print('\n ****** Linear Regression ******\n')

  # X_train, X_test, y_train, y_test are arrays having the k-fold datasets. Counter is the number of k that are used
  X_train, X_test, y_train, y_test, counter = prepareDatasetKFold(interactionArray)
  # Initialize precision, accuracy and f1-score as empty arrays
  precision, accuracy, f1_score = [], [], []

  # For each k in k-fold, fit the model, predict, and calculate metrics
  for i in range(counter):
    # Original dataset shape
    # print('Original dataset shape %s' % Counter(y_train[i]))

    # Using BorderlineSMOTE for upsampling the dataset
    sm = BorderlineSMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train[i], y_train[i])

    # Resampled dataset shape
    # print('Resampled dataset shape %s' % Counter(y_train_res))

    # Train the model using fit() with X_train_res and y_train_res
    model = LinearRegression().fit(X_train_res, y_train_res) 
    # Predict a Response
    y_pred = model.predict(X_test[i])
    y_pred = np.where(y_pred > 0, 1, 0)
    #Calculate the metrics using functions in sklearn
    precision.append(metrics.precision_score(y_test[i], y_pred))
    accuracy.append(metrics.accuracy_score(y_test[i], y_pred))
    f1_score.append(metrics.f1_score(y_test[i], y_pred))
  
  # Print accuracy, precision and f1-scores by taking the mean of the respective arrays
  print('Accuracy Score : ', np.mean(accuracy))
  print('F1 Score : ', np.mean(f1_score))

linearRegression(interactionArray)

