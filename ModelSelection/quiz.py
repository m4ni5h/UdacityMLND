# Import, read, and split data
import pandas as pd
data = pd.read_csv('data.csv')
import numpy as np
X = np.array(data[['x1', 'x2']])
y = np.array(data['y'])

# Fix random seed
np.random.seed(55)

### Imports
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC

# TODO: Uncomment one of the three classifiers, and hit "Test Run"
# to see the learning curve. Use these to answer the quiz below.

### Logistic Regression
# This is an example of underfitting 
estimator = LogisticRegression()

### Decision Tree
# This is a good fitting
estimator = GradientBoostingClassifier()

### Support Vector Machine
# This is an example of overfitting
estimator = SVC(kernel='rbf', gamma=1000)