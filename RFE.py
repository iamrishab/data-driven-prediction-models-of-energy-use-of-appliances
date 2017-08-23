#Recursive Feature Elimination
from sklearn import datasets
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
import pandas as pd
import numpy as np
a=["date","Appliances","T1","RH_1","T2","RH_2","T3","RH_3","T4","RH_4","T5","RH_5","T6","RH_6","T7","RH_7","T8","RH_8","T9","RH_9","T_out","RH_out","Windspeed","Visibility","rv1","rv2"]
#load the datasets
df = pd.read_csv(r"C:\Users\T00538\Desktop\training.csv",usecols=a,nrows=677)
df["date1"]=(df['date'].str.split(':').str[0])
del a[0]
table = pd.pivot_table(df,index=["date1"],
               values = a,
               aggfunc=[np.sum],fill_value=0)


y = np.array(table[("sum",'Appliances')])
del table[("sum","Appliances")]
X = np.array(table.values.tolist())



#create a base classifier used to evaluate a subset of attributes
model=SVR(kernel='linear')

#create the RFE model and select 3 attributes
rfe=RFE(model,22)
rfe=rfe.fit(X,y)

#summarize the selection of the attributes
n=list(table.columns.get_level_values(1))
for i in range(len(a)-1):
    print(str(n[i])+" : "+str(rfe.support_[i])+", rank: "+str(rfe.ranking_[i]))

