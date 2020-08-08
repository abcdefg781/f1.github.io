import numpy as np
import pandas as pd
import plotly.express as px
from functools import reduce

df = pd.read_csv('./f1db_csv/lap_times.csv')
df_drivers = pd.read_csv('./f1db_csv/drivers.csv')
df2 = df[["raceId","driverId","lap","milliseconds"]]
df3 = df2[(df2["raceId"]==1034)]
df_merged = df3.merge(df_drivers[["driverId","driverRef"]],on='driverId')

df_laptimes = df_merged[['lap','driverRef',"milliseconds"]]

#start code for this graph

df_grouped = [y for x,y in df_laptimes.groupby('driverRef',as_index=False)]
for i in range(len(df_grouped)):
	df_driver = df_grouped[i]
	df_grouped[i] = pd.concat([df_driver[["lap","driverRef"]],df_driver["milliseconds"].cumsum()],axis=1)
	df_grouped[i] = df_grouped[i].reset_index(drop=True)

#get minimum cumulative time for each lap

df_merged = pd.concat(df_grouped)
print(df_merged)

df_minimum = [y for x,y in df_merged.groupby('lap',as_index=False)]

for i in range(len(df_minimum)):
	#df_minimum[i]['min'] = df_minimum[i]['milliseconds'].max()
	df_minimum[i]['min'] = df_minimum[i]['milliseconds'][df_minimum[i]['driverRef']=='bottas']
	df_minimum[i]['delta'] = df_minimum[i]['milliseconds']-df_minimum[i]['min']
	df_minimum[i] = df_minimum[i][['lap','driverRef','delta']]

df_deltas = pd.concat(df_minimum,ignore_index=True)
df_deltas['delta_seconds'] = df_deltas['delta']/1000

fig = px.line(df_deltas,x="lap",y="delta_seconds",color='driverRef')
fig.show()


# pd.set_option('display.max_columns', None)
# df_merged = reduce(lambda x,y: pd.merge(x,y, on='lap', how='outer'), df_grouped)
