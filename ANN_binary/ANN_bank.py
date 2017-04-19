# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 14:26:25 2017
Title: Demo - Deep learning
Subtitle: Artificial Neural Network for bank's costumer prediction
"""
# This demo demonstrates the ability of an Artificial Neural Network (ANN) 
# to generate preditions from a data set with several features and 
# highly non-linear behaviour. 
# A fake, but relalistic, bank's customer data set will be used to predict 
# if a customer will leave or not a bank using historical data with 
# several features like the number of products, credit score, 
# estimate salary, etc.

#This demo comes from an excellent [tutorial](https://www.superdatascience.com/deep-learning/) 
# from Kirill Eremenko and Hadelin de Ponteves.

# Requirements: pandas, keras with TensorFlow backend, sklearn libraries and 
# packages are required. This codes has been tested on a Windows 10 machine 
# and Python 3.5.

### Importing libraries
import pandas as pd


### Importing the data set
dataset = pd.read_csv('Churn_Modelling.csv')
dataset.shape

X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

### Data Pre-processing
#### Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Encode categorizes into numerical data:
# For Geography:
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

# For the new Geography column, it is needed to indicate to the algorithm 
# that the columns are factors and not numbers, so the so-called "dummy variables" 
# have to be generated:
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()

# To avoid the dummy variables trap, one dummy variable column is removed:
X = X[:,1:]

#### Preparing training and testing data sets
# For training and testing of the algorithm, the data set needs to be split:
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#### Scaling of the data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

### Creating an ANN with Keras
#### Importing the Keras libraries and packages
from keras.wrappers.scikit_learn import KerasClassifier # keras wrapper for k-validation
from sklearn.model_selection import cross_val_score # k-fold validation
from keras.models import Sequential # used to initiate the ANN
from keras.layers import Dense # used to create the layers of the ANN
from sklearn.model_selection import cross_val_score # k-fold validation
from keras.layers import Dropout # used to avoid overfitting

#### Building the ANN
def build_classifier():
    # Initialising the ANN
    classifier = Sequential()
    # Adding the input layer and the FIRST hidden layer
    classifier.add(Dense(units = 16, init = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dropout(rate = 0.1))
    # Adding the two more hidden layers
    for i in range(1):
        classifier.add(Dense(units = 16, init = 'uniform', activation = 'relu'))
        classifier.add(Dropout(rate = 0.1))
    # Adding the output layer
    classifier.add(Dense(units = 1, init = 'uniform', activation = 'sigmoid'))
    # Compiling the ANN
    classifier.compile(optimizer = 'sgd', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier, batch_size = 24, epochs=500)

#### Evaluating ANN with k-fold validation
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = 1)
mean = accuracies.mean()
variance = accuracies.std()

