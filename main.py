# Check the versions of libraries
import math
# Python version
import sys
print('Python: {}'.format(sys.version))
# scipy
import scipy
print('scipy: {}'.format(scipy.__version__))
# numpy
import numpy
print('numpy: {}'.format(numpy.__version__))
# matplotlib
import matplotlib
print('matplotlib: {}'.format(matplotlib.__version__))
# pandas
import pandas
print('pandas: {}'.format(pandas.__version__))
# scikit-learn
import sklearn
print('sklearn: {}'.format(sklearn.__version__))
# Load libraries
import pandas
#from pandas.tools.plotting import scatter_matrix
from pandas import to_datetime
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.externals import joblib
import entries
import platform
import news_scraper

# Load dataset
url="C:\\Users\\John\\Desktop\\StkPreSys2018\\TSLA2.csv"

if(platform.system() =='Windows'):
	pass
else:
	new_url  = input("Enter a stock code:")
	url =  entries.lookup(new_url)


#names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
# Load dataset
dataset = pandas.read_csv(url)
# shape
print(dataset.shape)
print(dataset.head(20))
print(dataset.describe())
# Split-out validation dataset
array = dataset.values
X = array[:,0:8]
Y = array[:,8]
my_dates = array[:,0]
adapted_dates = to_datetime(my_dates, format="%Y%m%d")
print(X)
validation_size = 0.30
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
# Spot Check Algorithms
models = []
models.append(('Lasso', Lasso(max_iter=10000)))
models.append(('ElasticNet', ElasticNet(max_iter=10000)))
models.append(('AdaBoost', AdaBoostRegressor()))
models.append(('GradientBoost', GradientBoostingRegressor()))
models.append(('Ridge', Ridge(max_iter=10000)))
# Test options and evaluation metric
seed = 8
scoring = 'neg_mean_absolute_error'
# evaluate each model in turn
results = []
names = []
d={}
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    d.update({model:abs(cv_results.mean())})
    print(msg)
print(d) 
# Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
# Make predictions on validation dataset
lpred = []
model = min(d, key=d.get)
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)
for row in predictions:
    lpred.append(row)
print(Y_validation)
print(lpred)
print("r2: %f" % r2_score(Y_validation, predictions))
print("mse: %f" % mean_squared_error(Y_validation, predictions))
print("ev: %f" % explained_variance_score(Y_validation, predictions))
import matplotlib.pyplot as plt
Ysize=int(Y.shape[0]*validation_size)
plt.plot(adapted_dates[-Ysize-1:], Y_validation)
plt.plot(adapted_dates[-Ysize-1:], lpred)
plt.legend(['Y_Validation', 'lpred'], loc='upper left')
plt.show()
plt.figure()
print(model)
sentiment = news_scraper.scrape("Tesla", category="business")
#print('Sentiment: ', sentiment)
#Testing the validation model in a practical setting
joblib.dump(model, 'pythia.pkl')
joblib.load('pythia.pkl')
#Create prediction of all of the closes.
y_pred = []
y_predictions = model.predict(X)
for i in y_predictions:
    y_pred.append(i)
#Plotting
Y_pred_size=int(Y.shape[0])
plt.plot(adapted_dates[-Y_pred_size:], Y)
plt.plot(adapted_dates[-Y_pred_size:], y_pred)
plt.legend(['Y_Validation', 'y_pred'], loc='upper left')
plt.show()
plt.figure()
#Create array
import pandas as pd
data_array_1 = {'date': adapted_dates[-Ysize-1:], 'Y_validation': Y_validation, 'lpred': lpred}
df = pd.DataFrame(data=data_array_1)
print(df.head())
#Write to csv
outfile = open('output.csv', 'w')
df.to_csv('output.csv')
outfile.close()