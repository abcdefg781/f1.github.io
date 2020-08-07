import numpy as np
import pandas as pd
import plotly.express as px

df = pd.read_csv('./f1db_csv/lap_times.csv')
df_drivers = pd.read_csv('./f1db_csv/drivers.csv')



df2 = df[["raceId","driverId","lap","milliseconds"]]

df3 = df2[(df2["raceId"]==1034)]
df_merged = df3.merge(df_drivers[["driverId","driverRef"]],on='driverId')
print(df_merged)

fig = px.line(df_merged,x='lap',y='milliseconds',color='driverRef')

fig.show()