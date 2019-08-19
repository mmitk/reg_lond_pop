import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
from scipy.interpolate import *
#%matplotlib inline

df = pd.read_csv("london_pop.csv")

#take a look at the dataset
#df.head()

# summarize the data
df.describe()

#selecting various features
cdf = df[['ActualYear','Value']]
plt.scatter(cdf.ActualYear, cdf.Value,  color='blue')
plt.xlabel("Year")
plt.ylabel("Value")
plt.show()


#Function for plotting a generic parabola with given parameters 
def parabola(x, a,b,c):
    return a*x**2 + b*x + c


#Train/Test split: (80/20)
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]
x_test =test['ActualYear']
y_test =test['Value']

#######################################################
#training a polyfit model with a second degree polynomial
x_train =train['ActualYear']
y_train =train['Value']
p1 = np.polyfit(x_train,y_train,2)
y_hat = parabola(x_test, p1[0],p1[1],p1[2])


xp = np.linspace(1950,65,2015)

plt.scatter(cdf.ActualYear, cdf.Value,  color='red')
plt.xlabel("Year")
plt.ylabel("Value")
plt.plot(xp,np.polyval(p1,xp),'b--')
plt.show()
from sklearn.metrics import r2_score

print('Using Polyfit:.........................')
print(" a = %f, b = %f, c=%f" % (p1[0], p1[1],p1[2]))
print("Mean absolute error: %.2f" % np.mean(np.absolute(y_hat - y_test)))
print("Residual sum of squares: %.2f" % np.mean((y_hat - y_test) ** 2))
print("R2-score: %.2f" % r2_score(y_hat , y_test) )



#curve fit uses non-linear least squares to fit parabola function
#to the data. popt are the optimized parameters
from scipy.optimize import curve_fit
popt, pcov = curve_fit(parabola, x_train, y_train)

######################################################################################
#Using curve_fit rather than polyfit
y = parabola(x_train, *popt)
#plotting data
plt.plot(x_train, y_train, 'ro', label='data')
#plotting sigmoid against the data
plt.plot(x_train,y, linewidth=3.0, label='fit')
plt.legend(loc='best')
plt.ylabel('Population')
plt.xlabel('Year')
plt.show()

# predict using test set
y_hat = parabola(x_test, *popt)


# evaluation
#print the final parameters
print('Using Curve Fit:....................................................')
print(" a = %f, b = %f, c=%f" % (popt[0], popt[1],popt[2]))
print("Mean absolute error: %.2f" % np.mean(np.absolute(y_hat - y_test)))
print("Residual sum of squares: %.2f" % np.mean((y_hat - y_test) ** 2))
print("R2-score: %.2f" % r2_score(y_hat , y_test) )
