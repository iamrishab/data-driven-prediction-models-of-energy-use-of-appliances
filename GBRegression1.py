

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import ensemble
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error

#Load Data
a=["date","Appliances","T1","RH_1","T2","RH_2","T3","RH_3","T4","RH_4","T5","RH_5","T6","RH_6","T7","RH_7","T8","RH_8","T9","RH_9","NSM","T_out","RH_out","Windspeed","Visibility","rv1","rv2"]
#load the datasets

#raw_data = urllib.urlopen(r"C:\Users\T00538\Desktop\training.csv")
# load the CSV file as a numpy matrix
#dataset = np.loadtxt(raw_data, delimiter=",")

df = pd.read_csv(r"C:\Users\T00538\Desktop\training.csv",usecols=a, nrows = 677)
df1 = pd.read_csv(r"C:\Users\T00538\Desktop\testing_validation.csv",usecols=a, nrows=677)
df1["date1"]=(df1['date'].str.split(':').str[0])
df["date1"]=(df['date'].str.split(':').str[0])
del a[0]


table = pd.pivot_table(df,index=["date1"],
               values = a,
               aggfunc=[np.mean],fill_value=0)
table1 = pd.pivot_table(df1,index=["date1"],
               values = a,
               aggfunc=[np.mean],fill_value=0)

y = np.array(table[("mean",'Appliances')])
y_test = np.array(table1[("mean",'Appliances')])
del table[("mean","Appliances")]
del table1[("mean","Appliances")]
X = np.array(table.values.tolist())
X_test = np.array(table1.values.tolist())

#Fit Regression Model

params={'n_estimators':500,'max_depth':4,'min_samples_split':2,
        'learning_rate':0.01,'loss':'ls'}
clf=ensemble.GradientBoostingRegressor(**params)

clf.fit(X,y)
mse=mean_squared_error(y_test,clf.predict(X_test))
print("MSE:%.4f" % mse)

#Plot Training Deviance

# compute test set deviance
test_score=np.zeros((params['n_estimators'],), dtype=np.float64)

for i,y_pred in enumerate(clf.staged_predict(X_test)):
    test_score[i]=clf.loss_(y_test,y_pred)

plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.title('Deviance')

plt.plot(np.arange(params['n_estimators'])+1, clf.train_score_,'b-',
         label='Training Set Deviance')
plt.plot(np.arange(params['n_estimators'])+1,test_score,'r-',
         label='Test Set Deviance')
plt.legend(loc='upper right')
plt.xlabel('Boosting Iterations')
plt.ylabel('Deviance')

#Plot Feature Importance
feature_importance=clf.feature_importances_
# make importances relative to max importance
feature_importance=100.0*(feature_importance/feature_importance.max())
##sorted_idx=np.argsort(feature_importance, order = None)
##sorted_idx=np.array(feature_importance).argsort()[::-1]
sorted_idx = np.argsort(feature_importance)
ypos=[]
del a[0]
for i in sorted_idx :
    ypos.append(a[i])
pos=np.arange(sorted_idx.shape[0])+.5
plt.subplot(1,2,2)
plt.barh(pos,feature_importance[sorted_idx],align='center')
plt.yticks(pos,ypos)
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()
