from sklearn.preprocessing import MinMaxScaler
import numpy

weights = numpy.array([[115.0],[140.0],[175.0]])
scaler =  MinMaxScaler()
rescale_weight = scaler.fit_transform(weights)
print (rescale_weight)