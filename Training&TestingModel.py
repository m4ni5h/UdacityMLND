#Loading Data into Pandas

import pandas

# TODO: Use pandas to read the '2_class_data.csv' file, and store it in a variable
# called 'data'.
data = pandas.read_csv("2_class_data.csv")

#---------------------------------------------------------------------------------------------------
#Numpy Arrays
import pandas as pd
import numpy as np

data = pd.read_csv("data.csv")

# TODO: Separate the features and the labels into arrays called X and y

X = np.array(data[['x1','x2']])
y = np.array(data['y'])

#---------------------------------------------------------------------------------------------------
#Training models in sklearn
import pandas
import numpy

# Read the data
data = pandas.read_csv('data.csv')

# Split the data into X and y
X = numpy.array(data[['x1', 'x2']])
y = numpy.array(data['y'])

# import statements for the classification algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# TODO: Pick an algorithm from the list:
# - Logistic Regression
classifier = LogisticRegression()
classifier.fit(X,y)

# - Decision Trees
classifier = DecisionTreeClassifier()
classifier.fit(X,y)

# - Support Vector Machines
classifier = SVC()
classifier.fit(X,y)
# Define a classifier (bonus: Specify some parameters!)
# and use it to fit the data, make sure you name the variable as "classifier"
# Click on `Test Run` to see how your algorithm fit the data!
