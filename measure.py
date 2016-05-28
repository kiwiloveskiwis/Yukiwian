import numpy as np
import scipy as sp
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline


def rmse(y_ex,y):
	return np.sqrt(np.mean((y_ex-y)**2))
	
def comp(y_ex,y):
	y_mean=np.array(y)
	y_mean[:]=np.mean(y_mean)
	return 1-rmse(y_ex,y)/rmse(y_mean,y)

def measure(y):
	x=np.linspace(1,183,183)
	y_ex=[]
	y_ex=np.array(y_ex)
	
	pred=Pipeline([('poly',PolynomialFeatures(10)),
				   ('linear',LinearRegression(fit_intercept=False))])
	pred.fit(x[:,np.newaxis],y)	
	y_ex=pred.predict(x[:,np.newaxis])
	
	t=comp(y_ex,y)
    return t
	