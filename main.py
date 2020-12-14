# Check the versions of libraries

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

# Load dataset
url = "TSLA1.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
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
models.append(('Lasso', Lasso()))
models.append(('ElasticNet', ElasticNet()))
models.append(('SVR', SVR()))
models.append(('AdaBoost', AdaBoostRegressor()))
models.append(('GradientBoost', GradientBoostingRegressor()))
models.append(('SVR', SVR()))
models.append(('Ridge', Ridge()))
# Test options and evaluation metric
seed = 7
scoring = 'neg_mean_absolute_error'
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
# Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
# Make predictions on validation dataset
lpred = []
lasso = Lasso()
lasso.fit(X_train, Y_train)
predictions = lasso.predict(X_validation)
for row in predictions:
    lpred.append(row)
print(Y_validation)
print(lpred)
print("r2: %f" % r2_score(Y_validation, predictions))
print("mse: %f" % mean_squared_error(Y_validation, predictions))
print("ev: %f" % explained_variance_score(Y_validation, predictions))
import matplotlib.pyplot as plt
plt.plot(adapted_dates[-377:], Y_validation)
plt.plot(adapted_dates[-377:], lpred)
plt.legend(['Y_Validation', 'lpred'], loc='upper left')
plt.show()
plt.figure()