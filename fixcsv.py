#fix csv issues (red flags etc)

#lap times
import csv
import numpy as np
import pandas as pd

lap_times_df = pd.read_csv('./f1db_csv/lap_times.csv')

lap_times_byrace = [y for x,y in lap_times_df.groupby('raceId',as_index=False)]

for i in range(len(lap_times_byrace)):
	lap_times_bylap = [y for x,y in lap_times_byrace[i].groupby('lap',as_index=False)]
	for j in range(len(lap_times_bylap)):
		lap_times_bylap[j].loc[lap_times_bylap[j]['milliseconds']>400000,'milliseconds'] = 0
		if len(lap_times_bylap[j].loc[lap_times_bylap[j]['milliseconds']>400000,'milliseconds'])>0:
			print(i,j)
			print(lap_times_bylap[j].loc[lap_times_bylap[j]['milliseconds']>400000])

	lap_times_byrace[i] = pd.concat(lap_times_bylap)
lap_times_df = pd.concat(lap_times_byrace)
lap_times_df.to_csv('./f1db_csv/lap_times.csv')