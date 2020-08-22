import pandas as pd
import numpy as np
import copy
import datetime as dt
import warnings

warnings.filterwarnings("ignore")

drivers_df = pd.read_csv("./f1db_csv/drivers.csv").drop(columns = ['url','dob','nationality','number','code'])
races_df = pd.read_csv("./f1db_csv/races.csv").drop(columns=['time','url','circuitId'])
standings_df = pd.read_csv("./f1db_csv/driver_standings.csv").drop(columns=['wins','position','positionText'])
results_df = pd.read_csv("./f1db_csv/results.csv")
constructor_colors_df = pd.read_csv("./f1db_csv/constructors_colors.csv")

races_df['date'] = pd.to_datetime(races_df['date'])

merged_df = standings_df.merge(races_df,on='raceId')
merged_df = merged_df.merge(drivers_df,on='driverId')

merged_df = merged_df[merged_df['year']==2020]
merged_df.drop(['driverStandingsId'],axis=1,inplace=True)

merged_df = merged_df.merge(results_df,on=['raceId','driverId'])
merged_df = merged_df.merge(constructor_colors_df,on='constructorId')

print(merged_df.iloc[0])
