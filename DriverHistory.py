#!/usr/bin/env python
# coding: utf-8

# # Driver History Tables

# In[1]:


import pandas as pd
import copy
import warnings
import datetime as dt
import numpy as np

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


# In[2]:


# Import all the data
drivers_df = pd.read_csv("./f1db_csv/drivers.csv").drop(columns = "url")
results_df = pd.read_csv("./f1db_csv/results.csv")
constructors_df = pd.read_csv("./f1db_csv/constructors.csv")
races_df = pd.read_csv("./f1db_csv/races.csv")
qualifying_df = pd.read_csv("./f1db_csv/qualifying.csv")
d_standings_df = pd.read_csv("./f1db_csv/driver_standings.csv")

# Clean some names and create new variables
# drivers_df
drivers_df["number"] = drivers_df["number"].replace({r"\N": None})
drivers_df["driverName"] = drivers_df["forename"].str.cat(drivers_df["surname"],sep = " ")
drivers_df = drivers_df.drop(columns = ["forename", "surname"])

# results_df
results_df["position"] = results_df["position"].replace({r"\N": None})

# Merging to a full dataset for driver history
df_1 = pd.merge(drivers_df[["driverId", "driverName", "number", "nationality"]], results_df[["driverId", "raceId", "constructorId", "position", "fastestLapTime"]], on = "driverId")
df_2 = pd.merge(df_1, constructors_df[["constructorId", "name"]], on = "constructorId")
df_2 = df_2.rename({"position" : "racePosition"}, axis = "columns")
df_2 = df_2.rename({"name" : "constructorName"}, axis = "columns")
df_3 = pd.merge(df_2, d_standings_df[["driverId", "raceId", "points", "position", "wins"]], on = ["driverId", "raceId"])
df_3 = df_3.rename({"position" : "driverStanding"}, axis = "columns")
df_4 = pd.merge(df_3, races_df[["raceId", "year", "name", "date"]], on = "raceId")
df_5 = pd.merge(df_4, qualifying_df[["raceId", "driverId", "position", "q1", "q2", "q3"]], on = ["driverId", "raceId"])

for i in range(len(df_5['q1'])):
    nan_series = df_5.q1.isna()[i]
    if (df_5['q1'][i] == r"\N") | (nan_series == True):
        df_5['q1'][i] = None
        i += 1
    elif df_5['q1'][i] != 0:
        df_5['q1'][i] = float(str(df_5['q1'][i]).split(':')[1]) + (60 * float(str(df_5['q1'][i]).split(':')[0]))
        i += 1
    else:
        df_5['q1'][i] = None
        i += 1

for i in range(len(df_5['q2'])):
    nan_series = df_5.q2.isna()[i]
    if (df_5['q2'][i] == r"\N") | (nan_series == True):
        df_5['q2'][i] = None
        i += 1
    elif df_5['q2'][i] != 0:
        df_5['q2'][i] = float(str(df_5['q2'][i]).split(':')[1]) + (60 * float(str(df_5['q2'][i]).split(':')[0]))
        i += 1
    else:
        df_5['q2'][i] = None
        i += 1

for i in range(len(df_5['q3'])):
    nan_series = df_5.q3.isna()[i]
    if (df_5['q3'][i] == r"\N") | (nan_series == True):
        df_5['q3'][i] = None
        i += 1
    elif df_5['q3'][i] != 0:
        df_5['q3'][i] = float(str(df_5['q3'][i]).split(':')[1]) + (60 * float(str(df_5['q3'][i]).split(':')[0]))
        i += 1
    else:
        df_5['q3'][i] = None
        i += 1
        
for i in range(len(df_5['fastestLapTime'])):
    nan_series = df_5.fastestLapTime.isna()[i]
    if (df_5['fastestLapTime'][i] == r"\N") | (nan_series == True):
        df_5['fastestLapTime'][i] = None
        i += 1
    elif df_5['fastestLapTime'][i] != 0:
        df_5['fastestLapTime'][i] = float(str(df_5['fastestLapTime'][i]).split(':')[1]) + (60 * float(str(df_5['fastestLapTime'][i]).split(':')[0]))
        i += 1
    else:
        df_5['fastestLapTime'][i] = None
        i += 1

df_5["minQualifyingTime"] = df_5[["q1", "q2", "q3"]].min(skipna = True, axis = 1)
df_5 = df_5.drop(columns = ["q1", "q2", "q3"])


# In[3]:


df_minlap = pd.read_csv("./f1db_csv/min_laps.csv")
df_5 = df_5.merge(df_minlap,on='raceId')
df_5[df_5.year == 2020]


# In[4]:


df_5['race_lap_ratio'] = df_5['fastestLapTime']/df_5['minOverallRaceLap']
df_5['quali_lap_ratio'] = df_5['minQualifyingTime']/df_5['minOverallQualiLap']
# Turn date into datetime
df_5["date"] = pd.to_datetime(df_5["date"])


# In[5]:


# Clean this dataset: drop variables and rearrange
df = copy.deepcopy(df_5)
df = df.drop(columns = ["driverId", "constructorId"])
df = df[["driverName", "number", "nationality", "year", "name", "date","raceId", "constructorName", "position", "minQualifyingTime", "racePosition", "fastestLapTime", "wins", "points", "driverStanding"]]

# Turn date into datetime
df["date"] = pd.to_datetime(df["date"])

# Save it into a csv
df.to_csv("./f1db_csv/driver_history.csv")
