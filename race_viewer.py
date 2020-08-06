import csv
import numpy as np
import matplotlib.pyplot as plt

race_data = np.array([])
with open('./f1db_csv/lap_times.csv',newline='') as csvfile:
	next(csvfile)
	reader = csv.reader(csvfile,delimiter=',')
	for row in reader:
		if int(row[0])==1034:
			rowData = np.array(row[1:4]+[row[-1]]).astype(int)
			if len(race_data) == 0:
				race_data = rowData
				race_data = np.reshape(race_data,(1, race_data.size))
			else:
				race_data = np.vstack([race_data,rowData])

drivers = np.unique(race_data[:,0])
drivernames = np.empty(len(drivers),dtype='U50')

with open('./f1db_csv/drivers.csv',newline='') as csvfile:
	next(csvfile)
	reader = csv.reader(csvfile,delimiter=',')
	for row in reader:
		for i in range(len(drivers)):
			if int(row[0])==drivers[i]:
				drivernames[i]=row[1]

def getDriverLapData(race_data,driver):
	data = race_data[race_data[:,0]==driver]
	return data

def getDriverDelta(race_data,driver1,driver2):
	data1 = getDriverLapData(race_data,driver1)
	data2 = getDriverLapData(race_data,driver2)

	delta = (data1[:,3]-data2[:,3])/1000
	return delta

for i in range(len(drivers)):
	if drivernames[i]=='albon' or drivernames[i]=='max_verstappen' or drivernames[i]=='leclerc':
		data = getDriverLapData(race_data,drivers[i])

		plt.plot(data[:,1],data[:,3]/1000,label=drivernames[i])
plt.legend()
plt.ylim(88,95)
plt.show()

print(drivernames)
# delta = getDriverDelta(race_data,1,830)
# plt.plot(delta)
# plt.ylim()
# plt.show()

