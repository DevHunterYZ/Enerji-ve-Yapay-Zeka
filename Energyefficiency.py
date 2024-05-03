import numpy as np
import pandas as pd
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn') #set same style for all plots
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputRegressor
pd.options.mode.chained_assignment = None
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import randint as sp_randint
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor,BaggingRegressor,AdaBoostRegressor,BaggingRegressor
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
import keras

output = pd.DataFrame(index=None, columns=['model','train_r2_score','test_r2_score'])

data = pd.read_csv('ENB2012_data.csv')
data.head()
