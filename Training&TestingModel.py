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

#---------------------------------------------------------------------------------------------------
#Tuning Parameters Manually
import pandas
import numpy

# Read the data
data = pandas.read_csv('data.csv')

# Split the data into X and y
X = numpy.array(data[['x1', 'x2']])
y = numpy.array(data['y'])

# Import the SVM Classifier
from sklearn.svm import SVC

# TODO: Define your classifier.
# Play with different values for these, from the options above.
# Hit 'Test Run' to see how the classifier fit your data.
# Once you can correctly classify all the points, hit 'Submit'.
classifier = SVC(kernel = 'rbf', gamma = 200)

# Fit the classifier
classifier.fit(X,y)

#----------------------------------------------------------------------------------------------------------
#Testing your models
# Import statements 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

# Import the train test split
# http://scikit-learn.org/0.16/modules/generated/sklearn.cross_validation.train_test_split.html
from sklearn.cross_validation import train_test_split

# Read in the data.
data = np.asarray(pd.read_csv('data.csv', header=None))
# Assign the features to the variable X, and the labels to the variable y. 
X = data[:,0:2]
y = data[:,2]

# Use train test split to split your data 
# Use a test size of 25% and a random state of 42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Instantiate your decision tree model
model = DecisionTreeClassifier()

# TODO: Fit the model to the training data.
model.fit(X_train,y_train)

# TODO: Make predictions on the test data
y_pred = model.predict(X_test)

# TODO: Calculate the accuracy and assign it to the variable acc on the test data.
acc = accuracy_score(y_test, y_pred)