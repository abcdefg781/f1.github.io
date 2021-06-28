import pandas as pd
import numpy as np

print("test_function_mintime.py")

drivers_df = pd.read_csv("./f1db_csv/drivers.csv").drop(columns = "url")
lap_times_df = pd.read_csv("./f1db_csv/lap_times.csv")
results_df = pd.read_csv("./f1db_csv/results.csv")
constructors_df = pd.read_csv("./f1db_csv/constructors.csv")
races_df = pd.read_csv("./f1db_csv/races.csv")
colors_df = pd.read_csv("./f1db_csv/constructors_colors.csv")
driver_history_df = pd.read_csv("./f1db_csv/driver_history.csv")
qualifying_df = pd.read_csv("./f1db_csv/qualifying.csv")


minracelap_df_list = [y for x,y in lap_times_df.groupby('raceId',as_index=False)]

minracelap = []
for i in range(len(minracelap_df_list)):
    minlap = (minracelap_df_list[i]['milliseconds'].min())/1000 #in seconds
    minracelap.append([minracelap_df_list[i]['raceId'].iloc[0],minlap])

minracelap_df = pd.DataFrame(minracelap,columns=['raceId','minOverallRaceLap'])

minqualilap_df_list = [y for x,y in qualifying_df.groupby('raceId',as_index=False)]

minqualilap = []
for i in range(len(minqualilap_df_list)):
    race_quali_df = minqualilap_df_list[i]
    for i in range(len(race_quali_df['q1'])):
        nan_series = race_quali_df.q1.isna().iloc[i]
        if (race_quali_df['q1'].iloc[i] == r"\N") | (nan_series == True):
            race_quali_df['q1'].iloc[i] = None
            i += 1
        elif race_quali_df['q1'].iloc[i] != 0:
            race_quali_df['q1'].iloc[i] = float(str(race_quali_df['q1'].iloc[i]).split(':')[1]) + (60 * float(str(race_quali_df['q1'].iloc[i]).split(':')[0]))
            i += 1
        else:
            race_quali_df['q1'].iloc[i] = None
            i += 1

    for i in range(len(race_quali_df['q2'])):
        nan_series = race_quali_df.q2.isna().iloc[i]
        if (race_quali_df['q2'].iloc[i] == r"\N") | (nan_series == True):
            race_quali_df['q2'].iloc[i] = None
            i += 1
        elif race_quali_df['q2'].iloc[i] != 0:
            race_quali_df['q2'].iloc[i] = float(str(race_quali_df['q2'].iloc[i]).split(':')[1]) + (60 * float(str(race_quali_df['q2'].iloc[i]).split(':')[0]))
            i += 1
        else:
            race_quali_df['q2'].iloc[i] = None
            i += 1

    for i in range(len(race_quali_df['q3'])):
        nan_series = race_quali_df.q3.isna().iloc[i]
        if (race_quali_df['q3'].iloc[i] == r"\N") | (nan_series == True):
            race_quali_df['q3'].iloc[i] = None
            i += 1
        elif race_quali_df['q3'].iloc[i] != 0:
            race_quali_df['q3'].iloc[i] = float(str(race_quali_df['q3'].iloc[i]).split(':')[1]) + (60 * float(str(race_quali_df['q3'].iloc[i]).split(':')[0]))
            i += 1
        else:
            race_quali_df['q3'].iloc[i] = None
            i += 1

    minq1 = race_quali_df['q1'].min()
    minq2 = race_quali_df['q2'].min()
    minq3 = race_quali_df['q3'].min()

    #print(race_quali_df['raceId'].iloc[0],minq1,minq2,minq3,np.nanmin([minq1,minq2,minq3]))
    if race_quali_df['raceId'].iloc[0] == 256:
        minqualilap.append([256,75.505])
    else:
        minqualilap.append([race_quali_df['raceId'].iloc[0],np.nanmin([minq1,minq2,minq3])])

minqualilap_df = pd.DataFrame(minqualilap,columns=['raceId','minOverallQualiLap'])

minlap_df = minracelap_df.merge(minqualilap_df,on='raceId')
minlap_df.to_csv('./f1db_csv/min_laps.csv')