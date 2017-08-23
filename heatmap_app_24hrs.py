import pandas as pd
import numpy as np
import plotly.plotly as py
import plotly.graph_objs as go
#from pandas import DataFrame
import matplotlib.pyplot as plt
from matplotlib import cm

df = pd.read_csv(r"C:/Users/T00537/Desktop/Data Set/training.csv")
#df = df[df["date"] > "2016-1-12 00:00:00" & df["date"] < "2016-1-17 23:59:59"]
df=df[df["date"]< "2016-01-17 23:59:59"]
df["date1"]=(df['date'].str.split(':').str[0].str.split(" ").str[1])

arranged_day = pd.Categorical(df["Day_of_week"], categories=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday","Saturday","Sunday"],ordered=True)
day_series = pd.Series(arranged_day)

table = pd.pivot_table(df,index=["date1"],
               values="Appliances",columns=day_series,
               aggfunc=[np.sum],fill_value=0)


print(table)
fig, ax = plt.subplots()
ax.set_title('Heatmap : Appliances(wh)')

heatmap = ax.pcolor(table)
plt.colorbar(heatmap)

#cbar.ax.set_xticklabels(['< -1', '0', '> 1'])
#cbar.ax.set_yticklabels(table.columns.get_level_values(1))
#plt.pcolor(table)

#ax.set_yticks(np.arange(len(table.index)))
#ax.set_xticks(np.arange(len(table.columns)))

ax.set_yticks(range(len(table.index)+1))
ax.set_xticks(range(len(table.columns)+1))

ax.set_xticklabels(table.columns.get_level_values(1))

plt.xlabel("Week 2")
plt.ylabel("Hours of Day")
plt.show()
