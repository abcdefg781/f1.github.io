import pandas as pd
import numpy as np
import copy
import datetime as dt
import warnings

warnings.filterwarnings("ignore")

drivers_df = pd.read_csv("./f1db_csv/drivers.csv").drop(columns = "url")
lap_times_df = pd.read_csv("./f1db_csv/lap_times.csv")
results_df = pd.read_csv("./f1db_csv/results.csv")
constructors_df = pd.read_csv("./f1db_csv/constructors.csv")
races_df = pd.read_csv("./f1db_csv/races.csv")
colors_df = pd.read_csv("./f1db_csv/constructors_colors.csv")
driver_history_df = pd.read_csv("./f1db_csv/driver_history.csv")
qualifying_df = pd.read_csv("./f1db_csv/qualifying.csv")
minlap_df = pd.read_csv("./f1db_csv/min_laps.csv")
quali_form_df = driver_history_df.merge(minlap_df,on='raceId')

df_5 = copy.deepcopy(quali_form_df)
df_5['race_lap_ratio'] = df_5['fastestLapTime']/df_5['minOverallRaceLap']
df_5['quali_lap_ratio'] = df_5['minQualifyingTime']/df_5['minOverallQualiLap']
# Turn date into datetime
df_5["date"] = pd.to_datetime(df_5["date"])
df_grouped = [y for x,y in df_5.groupby('driverName',as_index=False)]

for i in range(len(df_grouped)):
    df_driver_grouped = [y for x,y in df_grouped[i].groupby('year',as_index=False)]
    for j in range(len(df_driver_grouped)):

        df_driver_grouped[j] = df_driver_grouped[j][['driverName','date','quali_lap_ratio']]
        df_driver_grouped[j].dropna(inplace=True)
        df_driver_grouped[j] = df_driver_grouped[j].sort_values(by='date')
        df_driver_grouped[j] = df_driver_grouped[j].reset_index(drop=True)

        for k in range(3):
            #df_driver_grouped[j].drop(['minOverallRaceLap','minOverallQualiLap','race_lap_ratio','number'],inplace=True,axis=1)
            
            if df_driver_grouped[j].empty:
                continue
            
            last_row = copy.deepcopy(df_driver_grouped[j].tail(1))
            curr_year = last_row.iloc[0]['date'].year
            first_day = dt.datetime(curr_year,1,1)
            
            days_in_year = []
            for l in range(len(df_driver_grouped[j])):
                date = df_driver_grouped[j].iloc[l].date
                days_in_year.append((date-first_day).days)
            df_driver_grouped[j]['days_in_year'] = days_in_year

            if len(df_driver_grouped[j])>1:
                #print(df_driver_grouped[j])
                if k == 0:
                    df = copy.deepcopy(df_driver_grouped[j])
                    df = df[df.quali_lap_ratio<=1.07]
                    df.reset_index(drop=True,inplace=True)
                    if df.empty or len(df)<5:
                        continue
                    fit = np.polyfit(x=df['days_in_year'],y=df['quali_lap_ratio'],deg=1)
                    linear = np.poly1d(fit)
                    df_driver_grouped[j]['fit_quali_lap_ratio_linear'] = linear(df_driver_grouped[j]['days_in_year'])
                    for l in range(len(df_driver_grouped[j]['fit_quali_lap_ratio_linear'])):
                        if df_driver_grouped[j]['fit_quali_lap_ratio_linear'][l]<1:
                            df_driver_grouped[j]['fit_quali_lap_ratio_linear'][l]=1.0
                elif k == 1:
                    df = copy.deepcopy(df_driver_grouped[j])
                    df = df[df.quali_lap_ratio<=1.07]
                    df.reset_index(drop=True,inplace=True)
                    if df.empty or len(df)<5:
                        continue
                    fit = np.polyfit(x=df['days_in_year'],y=df['quali_lap_ratio'],deg=2)
                    linear = np.poly1d(fit)
                    df_driver_grouped[j]['fit_quali_lap_ratio_quadratic'] = linear(df_driver_grouped[j]['days_in_year'])
                    for l in range(len(df_driver_grouped[j]['fit_quali_lap_ratio_quadratic'])):
                        if df_driver_grouped[j]['fit_quali_lap_ratio_quadratic'][l]<1:
                            df_driver_grouped[j]['fit_quali_lap_ratio_quadratic'][l]=1.0
                else:
                    df_driver_grouped[j]['fit_quali_lap_ratio_raw'] = df_driver_grouped[j]['quali_lap_ratio']

            elif len(df_driver_grouped[j]==1):
                df_driver_grouped[j]['fit_quali_lap_ratio_linear'] = df_driver_grouped[j]['quali_lap_ratio'].iloc[0]
                df_driver_grouped[j]['fit_quali_lap_ratio_quadratic'] = df_driver_grouped[j]['quali_lap_ratio'].iloc[0]
                df_driver_grouped[j]['fit_quali_lap_ratio_raw'] = df_driver_grouped[j]['quali_lap_ratio'].iloc[0]

        last_row.date = dt.datetime(curr_year,12,31)
        last_row.rolling_quali_lap_ratio = None
        last_row.quali_lap_ratio = None
        last_row.fit_quali_lap_ratio_linear = None
        last_row.fit_quali_lap_ratio_quadratic = None
        last_row.fit_quali_lap_ratio_raw = None
        last_row.days_in_year = None
        if j < len(df_driver_grouped)-1: #dont add the empty row when its the last year
            df_driver_grouped[j] = df_driver_grouped[j].append(last_row)

    df_grouped[i] = pd.concat(df_driver_grouped)
df_quali = pd.concat(df_grouped)
df_quali.to_csv('./f1db_csv/quali_form.csv')
