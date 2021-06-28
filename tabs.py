import time

# import matplotlib as mpl
# import matplotlib.cm

import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import numpy as np
import pandas as pd
import plotly.graph_objects as go
#import plotly.express as px
import plotly
from plotly import tools as tls
from dash.dependencies import Input, Output, State
import datetime as dt

# import smtplib
# from email.mime.multipart import MIMEMultipart
# from email.mime.text import MIMEText
# import sendgrid
# from sendgrid.helpers.mail import *
#import os
import copy
import warnings
import json

#fastf1 test
#fimport fastf1 as ff1
#ff1.Cache.enable_cache('./ff1cache')



#imports for lap sim
#from smt.surrogate_models import RBF
from RBF import RBF
#from sklearn.gaussian_process import GaussianProcessRegressor
#from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import joblib
#from linewidthhelper import linewidth_from_data_units
from spline import getSpline,getTrackPoints,getGateNormals2,reverseTransformGates,transformGates,displaceSpline,lerp
from Track import Track
import tracks


BS = "https://stackpath.bootstrapcdn.com/bootswatch/4.5.0/lux/bootstrap.min.css"
app = dash.Dash(__name__, external_stylesheets = [dbc.themes.BOOTSTRAP])
app.config['suppress_callback_exceptions'] = True
server = app.server
app.title = "Formulae 1"
warnings.filterwarnings("ignore")

app.index_string = """<!DOCTYPE html>
<html>
	<head>
		<!-- Global site tag (gtag.js) - Google Analytics -->
			<script async src="https://www.googletagmanager.com/gtag/js?id=UA-91329217-3"></script>
			<script>
			  window.dataLayer = window.dataLayer || [];
			  function gtag(){dataLayer.push(arguments);}
			  gtag('js', new Date());

			  gtag('config', 'UA-91329217-3');
			</script>
		{%metas%}
		<title>{%title%}</title>
		{%favicon%}
		{%css%}
	</head>
	<body>
		{%app_entry%}
		<footer>
			{%config%}
			{%scripts%}
			{%renderer%}
		</footer>
	</body>
</html>"""

################################################################
# Import data and Functions
# Import all the data

drivers_df = pd.read_csv("./f1db_csv/drivers.csv").drop(columns = "url")
lap_times_df = pd.read_csv("./f1db_csv/lap_times.csv",usecols=["raceId", "driverId", "lap", "milliseconds"])
results_df = pd.read_csv("./f1db_csv/results.csv")
constructors_df = pd.read_csv("./f1db_csv/constructors.csv")
races_df = pd.read_csv("./f1db_csv/races.csv")
races_df.sort_values(by=['year','raceId'],inplace=True,ascending=False)
driver_history_df = pd.read_csv("./f1db_csv/driver_history.csv")
constructor_colors_df = pd.read_csv("./f1db_csv/constructors_colors.csv")
standings_df = pd.read_csv("./f1db_csv/driver_standings.csv").drop(columns=['wins','position','positionText'])
raw_predictions_df = pd.read_csv("./predictions/ei_2020_predictions.csv").iloc[:, 1:]
# pr_predictions_df = pd.read_csv("./predictions/it_2020_pr_predictions.csv").iloc[:, 1:]

# Clean some names and create new variables
# drivers_df
drivers_df["number"] = drivers_df["number"].replace({r"\N": None})
drivers_df["driverName"] = drivers_df["forename"].str.cat(drivers_df["surname"],sep = " ")
drivers_df = drivers_df.drop(columns = ["forename", "surname"])

# lap_times_df
#lap_times_df = lap_times_df[["raceId", "driverId", "lap", "milliseconds"]]
lap_times_df["seconds"] = lap_times_df.milliseconds / 1000
lap_times_df.drop(columns = "milliseconds",inplace=True)

# results_df
results_df["position"] = results_df["position"].replace({r"\N": 99})
results_df["position"] = results_df["position"].astype(int)

# driver_history_df
driver_history_df["minQualifyingTime"] = round(driver_history_df.minQualifyingTime, 2)
driver_history_df["fastestLapTime"] = round(driver_history_df.fastestLapTime, 2)

#df_minlaps
minlap_df = pd.read_csv("./f1db_csv/min_laps.csv")

#quali form df
quali_form_df = pd.read_csv("./f1db_csv/quali_form.csv")
quali_form_df['date'] = pd.to_datetime(quali_form_df['date'])
#quali_form_df = driver_history_df.merge(minlap_df,on='raceId')
driver_history_df.drop(columns='raceId',inplace=True)

#rbf graph df
#rbf_df = pd.read_csv("./rbf_csv/rbfoutput.csv",header=None)
#s_df = pd.read_csv("./rbf_csv/s_bahrain.csv",header=None).to_numpy()
#yt = pd.read_csv("./rbf_csv/yt_bahrain.csv",header=None).to_numpy()
#xt = pd.read_csv("./rbf_csv/xt_bahrain.csv",header=None).to_numpy()



#slider values
lbounds = np.array([600000,2.0,0.8])
ubounds = np.array([1000000,6.0,2.0])
#normalizefactors = np.array([800000,4.0,1.4])
normalizefactors = np.array([1.0,1.0,1.0])

#RBF factor
sigma = 1.0

def create_race_table(year, race_name):
	races_temp = races_df[races_df.year == year]
	try:
		race_id = int(races_temp.raceId[races_temp.name == race_name])
	except TypeError:
		race_id = int(races_temp.raceId.iloc[0])
	lap_times_1 = lap_times_df[lap_times_df.raceId == race_id]
	results_1 = results_df[results_df.raceId == race_id]
	df_1 = pd.merge(drivers_df[["driverId", "driverName", "number"]], lap_times_1, on = "driverId")
	df_2 = pd.merge(df_1, results_1[["resultId", "driverId", "constructorId", "position"]], on = "driverId")
	df_3 = pd.merge(df_2, constructors_df[["constructorId", "constructorRef"]], on = "constructorId")
	df_3["constructorRef"] = df_3["constructorRef"].str.title()
	df_4 = pd.merge(df_3, races_df[["raceId", "year", "name"]], on = "raceId")
	df_4 = df_4.sort_values(by = ["position", "lap"])
	colored_df = pd.merge(df_4, constructor_colors_df[["constructorId", "color"]], on = "constructorId")
	color_palette = pd.Series(colored_df.color.values, index = colored_df.driverName).to_dict()
	return df_4, races_temp, color_palette, colored_df

def create_driver_table(driver_name_1):
	driver_1_table = driver_history_df[driver_history_df.driverName == driver_name_1].sort_values("date", ascending=False)
	driver_1_table = driver_1_table.drop(driver_1_table.columns[0:4], axis=1)
	return driver_1_table

class dataContainer:
	def __init__(self):
		self.race_table,self.year_races,self.color_palette,self.colored_df = create_race_table(2021, "Styrian Grand Prix")
		self.driver_history_table = create_driver_table("Lewis Hamilton")
		self.driver_yr_history_table = create_driver_table("Lewis Hamilton")
		
		self.num_functions = 3
		self.updateTrack(tracks.Bahrain_short)
	
		self.prevTrace = 0
		
	def updateTrack(self,track):
		self.track = track
		initial_direction = np.array([1,0])
		self.flipX = False
		self.flipY = False

		if track == tracks.Bahrain_short:
			sfile = "./rbf_csv/s_bahrain_short.csv"
			ytfile = "./rbf_csv/yt_bahrain_short.csv"
			xtfile = "./rbf_csv/xt_bahrain_short.csv"
			initial_direction = np.array([-1,0])
			self.trackWidth = 14
			trackname='bahrain_short'

		elif track == tracks.Bahrain:
			sfile = "./rbf_csv/s_bahrain.csv"
			ytfile = "./rbf_csv/yt_bahrain.csv"
			xtfile = "./rbf_csv/xt_bahrain.csv"
			initial_direction = np.array([-1,0])
			self.trackWidth = 14
			trackname='bahrain'
		elif track == tracks.Barcelona:
			sfile = "./rbf_csv/s_barcelona.csv"
			ytfile = "./rbf_csv/yt_barcelona.csv"
			xtfile = "./rbf_csv/xt_barcelona.csv"
			self.trackWidth = 10
			trackname='barcelona'
		elif track == tracks.AbuDhabi:
			sfile = "./rbf_csv/s_abudhabi.csv"
			ytfile = "./rbf_csv/yt_abudhabi.csv"
			xtfile = "./rbf_csv/xt_abudhabi.csv"

			self.trackWidth = 14
			initial_direction = np.array([0,-1])
			self.flipX = True
			trackname='abudhabi'

		elif track == tracks.Portimao:
			sfile = "./rbf_csv/s_portimao.csv"
			ytfile = "./rbf_csv/yt_portimao.csv"
			xtfile = "./rbf_csv/xt_portimao.csv"

			self.trackWidth = 14
			initial_direction = np.array([-1,0])
			self.flipY = True
			#self.flipX = True
			trackname='portimao'

		s_df = pd.read_csv(sfile,header=None).to_numpy()
		yt = pd.read_csv(ytfile,header=None).to_numpy()
		xt = pd.read_csv(xtfile,header=None).to_numpy()
		self.yt = yt
		self.xt = xt

		#normalize
		self.xt = self.xt/normalizefactors

		self.num_nodes = int(yt.shape[1]/self.num_functions)
		self.num_samples = yt.shape[0]

		#self.sm = RBF(lbounds,ubounds,sigma,xt,yt)

		#check if a pickled gaussian process regressor exists already
		picklepath = "./rbf_csv/"+trackname+".pkl"

		try:
			#self.sm = joblib.load(picklepath)
			self.sm = joblib.load(picklepath)
		except:
			#build surrogate model
			#kernel = C(1.0, (1e-3, 1e3)) * RBF(1, (1e-2, 1e2))
			#self.sm = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)

			#self.sm.fit(self.xt,self.yt)
			self.sm = RBF(lbounds/normalizefactors,ubounds/normalizefactors,sigma,xt,yt)

			#save pickle file
			joblib.dump(self.sm,picklepath)
			

		points = getTrackPoints(track,initial_direction)
		self.finespline,self.gates,self.gatesd,self.curv,slope = getSpline(points,s=0.0)

		self.trackLength = track.getTotalLength()

		normals = getGateNormals2(slope)

		self.normals = np.reshape(normals,(len(self.finespline[0]),2,2))


		#figure out aspect ratio of track
		dx = np.amax(self.finespline[0])-np.amin(self.finespline[0])
		dy = np.amax(self.finespline[1])-np.amin(self.finespline[1])
		self.dx = dx
		self.dy = dy
		self.aspect = dx/dy

		V = yt[:,0:self.num_nodes]
		self.trackColorScale = [np.amin(V)-1,np.amax(V)+1]
		self.trackDeltaColorScale = [-20,20]
		# self.s = s_df.iloc[0].to_numpy()
		self.s = s_df[0]

		x = (ubounds+lbounds)/2
		x = x/normalizefactors
		self.baseliney = self.sm.getGuess(x)
		#self.baseliney = self.sm.predict(np.atleast_2d(x))[0]


	def plotRaceGraph(self):
		#fig = px.line(self.race_table, x = "lap", y = "seconds", color = "driverName", hover_name = "driverName", hover_data = {"driverName" : False, "constructorRef" : True}, 
			#color_discrete_map = self.color_palette
			#)
		fig = go.Figure()
		# df_grouped = [y for x,y in self.race_table.groupby('driverName',as_index=False)]
		# for i in range(len(df_grouped)):
		#   driverName = df_grouped[i]['driverName'].iloc[0]
		#   fig.add_trace(go.Scatter(x = df_grouped[i]['lap'], y = df_grouped[i]['seconds'],mode='lines',name=driverName))
		

		df_team_grouped = [y for x,y in self.race_table.groupby('constructorRef',as_index=False)]
		avg_laptime_overall = self.race_table['seconds'].median()

		for i in range(len(df_team_grouped)):
			df_team_grouped[i].loc[df_team_grouped[i]['seconds']>300,'seconds'] = np.nan
			
			df_grouped = [y for x,y in df_team_grouped[i].groupby('driverName',as_index=False)]
			for j in range(len(df_grouped)):
				df_driver = df_grouped[j]
				avg_laptime = df_driver['seconds'].median()
				if self.filter == 1:
					df_driver.loc[df_driver['seconds']>1.02*avg_laptime,'seconds'] = np.nan
					df_driver.loc[df_driver['seconds']>1.2*avg_laptime_overall,'seconds'] = np.nan
				name = df_driver['driverName'].iloc[0]
				color = self.colored_df[self.colored_df.driverName==name]['color'].iloc[0]
				if j == 0:
					line = go.Scatter(x=df_driver['lap'],y=df_driver['seconds'],name=name,mode='lines',line=go.scatter.Line(color=color))
				else:
					line = go.Scatter(x=df_driver['lap'],y=df_driver['seconds'],name=name,mode='lines',line=go.scatter.Line(color=color,dash='dot'))
				fig.add_trace(line)

		#fig.update_layout(legend_title_text=None)

		fig.update_layout(plot_bgcolor="#323130",
			paper_bgcolor="#323130",font=dict(color="white"),
			xaxis_title="Lap",
			yaxis_title="Lap time (s)",
			margin = dict(l=20,r=20,t=20,b=20)
			)

		
		# fig.update_yaxes(automargin=True)
		return fig

	def plotRaceComparisonGraph(self, driver1, driver2):
		df_1 = self.race_table[self.race_table.driverName==driver1]["seconds"].rename('driver1')
		df_2 = self.race_table[self.race_table.driverName==driver2]["seconds"].rename('driver2')
		df_1 = df_1.reset_index(drop=True)
		df_2 = df_2.reset_index(drop=True)

		df_3 = pd.concat([df_1,df_2],axis=1)
		# df4['driver2'] = df_3[df_3.driverRef==driver2].seconds
		df_3['delta'] = df_3.driver2-df_3.driver1
		df_3['lap'] = df_3.index + 1

		df_1_summed = df_1.cumsum()
		df_2_summed = df_2.cumsum()
		df_3['cumdelta'] = df_2_summed-df_1_summed
		df_3['hovertext'] = [0]*len(df_3['cumdelta'])
		df_3['hovertext2'] = [0]*len(df_3['delta'])

		for i in range(len(df_3['cumdelta'])):
			time = '%.3f' % np.abs(df_3['cumdelta'][i])
			if df_3['cumdelta'][i]>0:
				df_3['hovertext'][i] = driver1+' is ahead of '+driver2+' by '+ time +' seconds'
			else:
				df_3['hovertext'][i] = driver1+' is behind '+driver2+' by '+time+' seconds'

		for i in range(len(df_3['delta'])):
			time = '%.3f' % np.abs(df_3['delta'][i])
			if df_3['delta'][i]>0:
				df_3['hovertext2'][i] = driver1+' is faster than '+driver2+' by '+ time +' seconds this lap'
			else:
				df_3['hovertext2'][i] = driver1+' is slower than '+driver2+' by '+time+' seconds this lap'

		df_3.loc[df_3['delta'].abs()>500,'delta'] = np.nan
		df_3.loc[df_3['cumdelta'].abs()>500,'cumdelta'] = np.nan


		fig = go.Figure()
		fig.add_trace(go.Scatter(x = df_3['lap'], y = df_3['delta'],mode='lines',name='Lap Delta',customdata=df_3['hovertext2'],hovertemplate=
			'<br>%{customdata}'))
		fig.add_trace(go.Scatter(x = df_3['lap'], y = df_3['cumdelta'],mode='lines',name='Cumulative Delta',customdata=df_3['hovertext'],hovertemplate=
			'<br>%{customdata}'))


		fig.update_layout(
			title={
				'text': 'Gap from '+driver1+' to '+driver2,
				'y':0.95,
				'x':0.5,
				'xanchor': 'center',
				'yanchor': 'top'},
			xaxis_title="Lap",
			yaxis_title="Delta (s)",
			hovermode="x unified",
			margin = dict(l=20,r=20,t=40,b=20)
			)

		fig.update_layout(plot_bgcolor="#323130",
			paper_bgcolor="#323130",font=dict(color="white"))
		fig.update_yaxes(automargin=True)
		return fig

	def plotDeltaGraph(self,deltaType):
		df_grouped = [y for x,y in self.race_table.groupby('driverName',as_index=False)]
		for i in range(len(df_grouped)):
			df_driver = df_grouped[i]
			df_grouped[i] = pd.concat([df_driver[["lap","driverName"]],df_driver["seconds"].cumsum()],axis=1)
			df_grouped[i] = df_grouped[i].reset_index(drop=True)

		#get minimum cumulative time for each lap

		df_merged = pd.concat(df_grouped)

		df_minimum = [y for x,y in df_merged.groupby('lap',as_index=False)]

		for i in range(len(df_minimum)):
			if deltaType == 'median':
				df_minimum[i]['min'] = df_minimum[i]['seconds'].median()
			elif deltaType == 'min':
				df_minimum[i]['min'] = df_minimum[i]['seconds'].min()
			elif deltaType == 'max':
				df_minimum[i]['min'] = df_minimum[i]['seconds'].max()
			else:
				df_minimum[i]['min'] = df_minimum[i]['seconds'][df_minimum[i]['driverName']==deltaType]

			df_minimum[i]['delta'] = df_minimum[i]['seconds']-df_minimum[i]['min']
			df_minimum[i] = df_minimum[i][['lap','driverName','delta']]

		df_deltas = pd.concat(df_minimum,ignore_index=True)
		df_deltas = df_deltas.merge(self.race_table[['driverName','constructorRef']],on='driverName')
		df_deltas = df_deltas.sort_values(by=["constructorRef","driverName"])

		df_team_grouped = [y for x,y in df_deltas.groupby('constructorRef',as_index=False)]

		fig = go.Figure()
		for i in range(len(df_team_grouped)):
			df_team_grouped[i].loc[df_team_grouped[i]['delta'].abs()>500,'delta'] = np.nan
			df_grouped = [y for x,y in df_team_grouped[i].groupby('driverName',as_index=False)]
			for j in range(len(df_grouped)):
				df_driver = df_grouped[j]
				name = df_driver['driverName'].iloc[0]
				color = self.colored_df[self.colored_df.driverName==name]['color'].iloc[0]
				if j == 0:
					line = go.Scatter(x=df_driver['lap'],y=df_driver['delta'],name=name,mode='lines',line=go.scatter.Line(color=color))
				else:
					line = go.Scatter(x=df_driver['lap'],y=df_driver['delta'],name=name,mode='lines',line=go.scatter.Line(color=color,dash='dot'))
				fig.add_trace(line)
		
		fig.update_layout(plot_bgcolor="#323130",
			paper_bgcolor="#323130",font=dict(color="white"),
			xaxis_title="Lap",
			yaxis_title="Delta (s)"
			)
		#fig.update_layout(xaxis=dict(range=[1,64.9]))
		if deltaType == 'min':
			titleString = 'Relative to first place'
		elif deltaType == 'max':
			titleString = 'Relative to last place'
		else:
			titleString = 'Relative to '+deltaType
		fig.update_layout(
			title={
				'text': titleString,
				'y':0.95,
				'x':0.5,
				'xanchor': 'center',
				'yanchor': 'top'})
		fig.update_layout(margin = dict(l=20,t=40,r=20,b=20))
		fig.update_yaxes(automargin=True,autorange="reversed")
		return fig

	def update_form_graph(self,chart_switch,quali_range):
		df_5 = quali_form_df[quali_form_df['date'].dt.year >= quali_range[0]]
		df_5 = df_5[df_5['date'].dt.year <= quali_range[1]]
		df_grouped = [y for x,y in df_5.groupby('driverName',as_index=False)]
		fig = go.Figure()
		
		for i in range(len(df_grouped)):
			name = df_grouped[i]['driverName'].iloc[0]
			if chart_switch == 0:
				line = go.Scatter(x=df_grouped[i]['date'],y=df_grouped[i]['fit_quali_lap_ratio_linear'],name=name,mode='lines')
			elif chart_switch == 1:
				line = go.Scatter(x=df_grouped[i]['date'],y=df_grouped[i]['fit_quali_lap_ratio_quadratic'],name=name,mode='lines')
			else:
				line = go.Scatter(x=df_grouped[i]['date'],y=df_grouped[i]['fit_quali_lap_ratio_raw'],name=name,mode='lines')
			fig.add_trace(line)

		fig.update_layout(plot_bgcolor="#323130",
				paper_bgcolor="#323130",font=dict(color="white"),
				xaxis_title="Date",
				yaxis_title="Qualifying lap time ratio",
				margin = dict(l=20,r=20,t=20,b=20)
				)
		fig.update_yaxes(automargin=True)
		return fig

	def plotStandingsGraph(self,year):
		merged_df = standings_df.merge(races_df,on='raceId')
		merged_df = merged_df.merge(drivers_df,on='driverId')

		year_races = races_df[races_df['year']==year]

		merged_df = merged_df[merged_df['year']==year]
		merged_df.drop(['driverStandingsId'],axis=1,inplace=True)
		results_df_drops = copy.deepcopy(results_df)
		results_df_drops.drop(['points'],axis=1,inplace=True)
		merged_df = merged_df.merge(results_df_drops,on=['raceId','driverId'])
		merged_df = merged_df.merge(constructor_colors_df,on='constructorId')
		grouped_df = [y for x,y in merged_df.groupby('constructorId',as_index=False)]

		
		fig = go.Figure()
		for i in range(len(grouped_df)):
			grouped_df2 = [y for x,y in grouped_df[i].groupby('driverId',as_index=False)]
			for j in range(len(grouped_df2)):
				grouped_df3 = grouped_df2[j]
				name = grouped_df3['driverName'].iloc[0]
				color = grouped_df3['color'].iloc[0]
				grouped_df3.sort_values(by='round',inplace=True)
				if j==0:
					line = go.Scatter(x=grouped_df3['round'],y=grouped_df3['points'],name=name,mode='lines',line=go.scatter.Line(color=color))
				elif j == 1:
					line = go.Scatter(x=grouped_df3['round'],y=grouped_df3['points'],name=name,mode='lines',line=go.scatter.Line(color=color,dash='dot'))
				elif j == 2:
					line = go.Scatter(x=grouped_df3['round'],y=grouped_df3['points'],name=name,mode='lines',line=go.scatter.Line(color=color,dash='dash'))
				else:
					line = go.Scatter(x=grouped_df3['round'],y=grouped_df3['points'],name=name,mode='lines',line=go.scatter.Line(color=color,dash='dashdot'))
				fig.add_trace(line)

		if len(grouped_df3['round'])==1:
			fig.update_traces(mode='markers', marker_line_width=0, marker_size=8)

		fig.update_layout(plot_bgcolor="#323130",
				paper_bgcolor="#323130",font=dict(color="white"),
				#xaxis_title="Date",
				yaxis_title="Championship Points",
				margin = dict(l=20,r=20,t=20,b=20)
				)
		fig.update_xaxes(ticktext=year_races['name'],tickvals=year_races['round'],tickangle=45)
		fig.update_yaxes(automargin=True)
		return fig
		

	def getDriverNames(self):
		return self.race_table.driverName.unique()

	def getRaceNames(self):
		today = pd.to_datetime('today')
		year_races = self.year_races[["raceId","year","name","date"]]
		year_races['datetimes'] = pd.to_datetime(year_races.date)
		year_races = year_races[year_races.datetimes<today]
		year_races = year_races.sort_values(by='datetimes',ascending=False)
		return year_races['name'].unique()

	def getAllDriverNames(self):
		return self.driver_history_table.driverName.unique()

	def updateRBF(self,x):
		
		x = x/normalizefactors
		# self.y = self.sm.predict(np.atleast_2d(x))[0]
		self.y = self.sm.getGuess(x)
		self.laptime = self.y[self.num_nodes*2-1]

	def plotTraceGraph(self,state):
		s = self.s

		if state == 0:
			ylabel = 'Velocity (km/h)'
		elif state == 1:
			ylabel = 'Time (s)'
		else:
			ylabel = 'Distance from centerline (m)'
		
		num_nodes = self.num_nodes

		fig = go.Figure()

		#Plot RBF data
		for i in range(self.num_samples):
			y = self.yt[i,num_nodes*(state):num_nodes*(state+1)]
			if state==0:
				y = y*3.6
			line = go.Scatter(x=s,y=y,opacity=0.2,line=dict(color='royalblue', width=1),hoverinfo='skip')
			fig.add_trace(line)

		#get ff1 data
		# if hasattr(self,'ff1data') == False:
			# self.ff1data = ff1.get_session(2020,'Portugal','Q')
			# self.laps = self.ff1data.load_laps(with_telemetry=True)
		# fast_ver = self.laps.pick_driver('HAM').pick_fastest()
		# ver_car_data = fast_ver.get_car_data()
		#print(ver_car_data.columns)
		# t_ver = ver_car_data['Time']
		# vCar_ver = ver_car_data['Speed']

		#RBF output
		# t = self.y[num_nodes:num_nodes*2]
		# t_ver = t_ver.dt.total_seconds()
		#print(t)
		y = self.y[num_nodes*(state):num_nodes*(state+1)]
		if state==0:
			y = y*3.6

		line = go.Scatter(x=s,y=y,mode='lines',line=dict(color='deepskyblue', width=4),name=ylabel)
		fig.add_trace(line)

		# line = go.Scatter(x=t_ver,y=vCar_ver,mode='lines',line=dict(color='orangered',width=4), name = 'ff1data')
		# fig.add_trace(line)

		
		fig.update_layout(plot_bgcolor="#323130",
			paper_bgcolor="#323130",font=dict(color="white"),
			xaxis_title="Distance along track (m)",
			yaxis_title=ylabel,
			showlegend=False,
			hovermode='x unified',
			margin = dict(l=20,r=20,t=20,b=20)

			)
		
		return fig

	def plotTrackGraph(self,absolute):
		y = self.y
		num_nodes = self.num_nodes
		func_index = 2
		n = y[num_nodes*(func_index):num_nodes*(func_index+1)]
		func_index = 0
		V = y[num_nodes*(func_index):num_nodes*(func_index+1)]
		V = V*3.6
		baselineV = self.baseliney[num_nodes*(func_index):num_nodes*(func_index+1)]
		baselineV = baselineV*3.6
		# if xaxisrange is not None:
		# 	markersize = np.maximum(15-((xaxisrange[1]-xaxisrange[0])*0.01),self.dx/400)
		# else:
		markersize = self.dx/400
		if absolute==0:
			line_array = self.plotTrackWithData(n,V,markersize=markersize,colormap='rainbow')
		else:
			line_array = self.plotTrackWithData(n,V,baselineV,markersize=markersize,colormap='rainbow')

		n1 = self.trackWidth/2 * np.ones(len(self.finespline[0]))
		n2 = -self.trackWidth/2 * np.ones(len(self.finespline[0]))

		displacedSpline1 = displaceSpline(n1,self.finespline,self.normals)
		displacedSpline2 = displaceSpline(n2,self.finespline,self.normals)

		if self.flipX == True:
			displacedSpline1[0] = -displacedSpline1[0]
			displacedSpline2[0] = -displacedSpline2[0]
		if self.flipY == True:
			displacedSpline1[1] = -displacedSpline1[1]
			displacedSpline2[1] = -displacedSpline2[1]

		#plot background track
		fig = go.Figure()
		line = go.Scatter(x=displacedSpline1[0],y=displacedSpline1[1],mode='lines',line=dict(color='white', width=2),hoverinfo='skip')
		fig.add_trace(line)
		#line = go.Scatter(x=displacedSpline1[0],y=displacedSpline1[1],mode='lines',line=dict(color='red', width=2,dash='dash'),hoverinfo='skip')
		#fig.add_trace(line)
		line = go.Scatter(x=displacedSpline2[0],y=displacedSpline2[1],mode='lines',line=dict(color='white', width=2),hoverinfo='skip')
		fig.add_trace(line)
		#line = go.Scatter(x=displacedSpline2[0],y=displacedSpline2[1],mode='lines',line=dict(color='red', width=2,dash='dash'),hoverinfo='skip')
		#fig.add_trace(line)

		#plot racing line
		for i in range(len(line_array)):
			fig.add_trace(line_array[i])

		

		fig.update_layout(plot_bgcolor="#323130",
			paper_bgcolor="#323130",font=dict(color="white"),
			showlegend=False,
			margin = dict(l=20,r=20,t=20,b=20),
			yaxis=dict(scaleanchor="x", scaleratio=1))
			#height=1000,
			#width=self.aspect*1000)

		# if xaxisrange is not None:
		# 	fig.update_xaxes(showgrid=False, zeroline=False,showticklabels=False,range=xaxisrange)
		# 	fig.update_yaxes(showgrid=False, zeroline=False,showticklabels=False,range=yaxisrange)
		# else:
		fig.update_xaxes(showgrid=False, zeroline=False,showticklabels=False)
		fig.update_yaxes(showgrid=False, zeroline=False,showticklabels=False)

		return fig

	def plotTrackWithData(self,n,state,baselinestate=None,markersize=4,colormap='rainbow'):
		s = self.s
		finespline = np.array(self.finespline)
		s_final = self.trackLength
		normals = self.normals
		newgates = []
		newnormals = []
		newn = []
		# for i in range(len(n)):
		# 	index = ((s[i]/s_final)*np.array(finespline).shape[1]).astype(int)
			
		# 	if index==np.array(finespline).shape[1]:
		# 		index = np.array(finespline).shape[1]-1
		# 	if i>0 and s[i] == s[i-1]:
		# 		continue
		# 	else:
		# 		newgates.append([finespline[0][index],finespline[1][index]])
		# 		newnormals.append(normals[index])
		# 		newn.append(n[i])

		index = ((np.unique(s)/s_final)*finespline.shape[1]).astype(int)
		if index[-1] == finespline.shape[1]:
			index[-1] = finespline.shape[1]-1
		newgates = np.array([finespline[0][index],finespline[1][index]]).T
		newnormals = normals[index]
		newn = n

		newgates = reverseTransformGates(newgates)
		displacedGates = displaceSpline(newn,newgates,newnormals)
		displacedGates = np.array((transformGates(displacedGates)))

		numFineSpline = 2000

		displacedSpline = getSpline(displacedGates,1/numFineSpline,0)[0]
		interp_state_array = np.zeros(len(displacedSpline[0]))

		s_new = np.linspace(0,s_final,numFineSpline)

		#plot spline with color
		line_array = []
		for i in range(1,len(displacedSpline[0])):
			# index = ((s_new[i]/s_final)*np.array(finespline).shape[1]).astype(int)
			s_spline = s_new[i]
			index_greater = np.argwhere(s>=s_spline)[0][0]
			index_less = np.argwhere(s<s_spline)[-1][0]

			x = s_spline
			if baselinestate is not None:
				interp_state = lerp(x,s[index_less],s[index_greater],state[index_less]-baselinestate[index_less],state[index_greater]-baselinestate[index_greater])
			else:
				interp_state = lerp(x,s[index_less],s[index_greater],state[index_less],state[index_greater])
			
			interp_state_array[i] = interp_state

			#print(index_less,index_greater,s[index_greater],s[index_less],s_spline,interp_state,fp[0],fp[1])
			# state_color = self.norm(interp_state)
			# color = self.cmap(state_color)
			# color = mpl.colors.to_hex(color)
			# point = [displacedSpline[0][i],displacedSpline[1][i]]
			# prevpoint = [displacedSpline[0][i-1],displacedSpline[1][i-1]]
			
			#line_array.append(go.Scatter(x=[point[0],prevpoint[0]],y=[point[1],prevpoint[1]],line=dict(color=color,width=1),hoverinfo='skip',mode='lines'))
		if baselinestate is not None:
			cmax = self.trackDeltaColorScale[1]*3.6
			cmin = self.trackDeltaColorScale[0]*3.6
		else:
			cmax = self.trackColorScale[1]*3.6
			cmin = self.trackColorScale[0]*3.6

		if self.flipX == True:
			displacedSpline[0] = -displacedSpline[0]
		if self.flipY == True:
			displacedSpline[1] = -displacedSpline[1]

		line_array.append(go.Scatter(x=displacedSpline[0],y=displacedSpline[1],marker=dict(
			size=markersize,
			cmax=cmax,
			cmin=cmin,
			color=interp_state_array,
			colorbar=dict(title="Velocity (km/h)"),
			colorscale=colormap),
			customdata=np.vstack((interp_state_array,s_new)).T,
			hovertemplate = 'Velocity: %{customdata[0]: .2f} km/h<br>Distance: %{customdata[1]:.2f} m<extra></extra>',
			mode='markers'))
		
		return line_array


#dataContainer object creation
dataContainer = dataContainer()

################################################################
# App Layout
app.layout = dbc.Container(
	[
		dcc.Location(id='url',refresh=False),
		dcc.Location(id='url-output',refresh=False),
		dcc.Store(id="store"),
		html.Br(),
		html.Br(),
		html.H1("f(1)"),
		dbc.Tabs(
			[
				dbc.Tab(label="Home", tab_id="home", children = [
						html.Br(),
						html.P("Welcome to formulae.one. The graph below shows the race for the driver's championship. Various analyses are available in the other tabs"),
						dcc.Dropdown(className='div-for-dropdown',id='standings_year',value=2021,clearable=False,options=[{'label': i, 'value': i} for i in races_df['year'].unique()]),
						dcc.Graph(className='div-for-charts',id='standings_graph',config={'toImageButtonOptions':{'scale':1,'height':800,'width':1700}},style={'width':'100%','height':'70vh','display':'flex','flex-direction':'column'})
					]),
				dbc.Tab(label="Race Analysis", tab_id="raceanalysis", children = [
						html.Br(),
						html.P('Select a year and race to view all charts on this page based on that race. The graph directly below shows the lap times in seconds for each driver as the race progresses. Slower lap times often indicate pit lane stops, safety cars, or incidents and drivers dropping out. These can be filtered out using the selection below.'),
						dbc.Row([
							dbc.Col(
								dcc.Dropdown(className='div-for-dropdown',id='year',value=2021,clearable=False,options=[{'label': i, 'value': i} for i in range(races_df['year'].max(),1995,-1)])
							),
							dbc.Col(
								dcc.Dropdown(className='div-for-dropdown',id='race_name',value='Styrian Grand Prix',clearable=False)
							)
						]),
						dbc.Row([
							dbc.Col(
								html.P("Filter out slow laps (pit stops/safety car/accidents/retirements):")),
							dbc.Col(
								dcc.Dropdown(className='div-for-dropdown',id='filter',value=0,clearable=False,options=[{'label': 'True', 'value': 1},{'label': 'False', 'value': 0}])
							),
							dbc.Col(),
							dbc.Col()
						]),
						dcc.Graph(className='div-for-charts',id='my-output',config={'toImageButtonOptions':{'scale':1,'height':800,'width':1700}}),
						html.Br(),
						html.P("The chart below shows the gap between all drivers relative to a reference driver or position. Select a driver, first/last place, or the median driver as the reference to compare gaps throughout the race."),
						dcc.Dropdown(className='div-for-dropdown',id='deltaType',value='min',clearable=False,searchable=False),
						dcc.Graph(className='div-for-charts',id='deltaGraph',config={'toImageButtonOptions':{'scale':1,'height':800,'width':1700}}),
						html.Br(),
						html.P("This final chart shows the difference between two chosen drivers for the selected race above. The lap delta is the difference in lap time for a given lap, and the cumulative delta shows the gap between the two drivers during the race up to that point."),
						dbc.Row([
							dbc.Col(
								dcc.Dropdown(className='div-for-dropdown',id='driver1',value='Lewis Hamilton',clearable=False)
							),
							dbc.Col(
								dcc.Dropdown(className='div-for-dropdown',id='driver2',value='Valtteri Bottas',clearable=False)
							)
						]),
						dcc.Graph(className='div-for-charts',id='my-output2',config={'toImageButtonOptions':{'scale':1,'height':800,'width':1700}})
					]),
				dbc.Tab(label = "Qualifying Form", tab_id="qualifying", children = [
						html.Br(),
						html.P("This chart shows trends in the qualifying performance of each driver for each season. The y-axis shows the ratio of their qualifying time to best lap time in the session. It is possible to view a linear or quadratic fit to the data, or to view the raw data for each race. Select a range of years in the slider below the graph."),
						dcc.Dropdown(className='div-for-dropdown',id='chart_switch', clearable=False,value=0,options=[{'label':'Linear fit','value':0},{'label':'Quadratic fit','value':1},{'label':'Raw data','value':2}]),
						dcc.Graph(className='div-for-charts',id='qualiFormGraph',config={'toImageButtonOptions':{'scale':1,'height':800,'width':1700}}),
						html.Br(),
						dcc.RangeSlider(id='quali_range',
							min=1996, max=2021, value=[2003, 2021],
							marks={
							1996: {'label': '1996'},
							1998: {'label': '1998'},
							2000: {'label': '2000'},
							2002: {'label': '2002'},
							2004: {'label': '2004'},
							2006: {'label': '2006'},
							2008: {'label': '2008'},
							2010: {'label': '2010'},
							2012: {'label': '2012'},
							2014: {'label': '2014'},
							2016: {'label': '2016'},
							2018: {'label': '2018'},
							2020: {'label': '2020'}
							})
				]),
				dbc.Tab(label = "Driver History", tab_id="driverhistory", children = [
						html.Br(),
						html.P("Select one driver to view their history. It is possible to search the dropdown menu"),
						dcc.Dropdown(className='div-for-dropdown',id='all_drivers', clearable=False,value='Lewis Hamilton'),    
						html.Div(id='my-table')
				]),
				dbc.Tab(label = "Race Predictions", tab_id="predictions", children =[
						dcc.Markdown('''
							## Predictions for the 2020 Eifel Grand Prix   
							The predictive model is updated every week after qualifying and before the race. The model uses an XGBoost algorithm to predict driver finishing position based on their qualifying performance, performance from previous races, driver and constructor standing, and weather.
						'''),
						# html.Br(),
						# dcc.Markdown('''
						#     *Predictions Based on Raw Data*  
						#     The model based on "raw" data, or the original variables from the F1 Ergast Data, predict the average lap time in a race for each driver based on current conditions such as weather and constructor, qualifying results, and results from the previous race. Because the raw data model relies so heavily on data from the race occuring right before it, instances where drivers DNF tend to perform poorly in the model.
						# '''),
						html.Br(),
						html.Div(children = [
							dash_table.DataTable(
							id = "raw_preds",
							columns=[{"name": i, "id": i} for i in raw_predictions_df.columns],
							data=raw_predictions_df.to_dict("rows"),
							style_table={'maxWidth': '100%'},
							style_header={'backgroundColor': 'rgb(30, 30, 30)', 
								#'whiteSpace': 'normal',
								'height': 'auto',
								#'width' : '20px'
							},
							style_cell={
								'backgroundColor': 'rgb(50, 50, 50)',
								'color': 'white',
								#'width' : '20px'
							},
							style_cell_conditional=[{
								'textAlign': 'center'
							}],
							style_as_list_view=True,
							),
						],
						style = {'width': '40%', 'margin' : 'auto'}
						),  
						# html.Br(),
						# dcc.Markdown('''
						#     *Predictions with Feature Engineering*  
						#     For the model with "processed" data, the qualifying results and fastest lap times were turned into ratios based on the driver with pole position for that race, and the driver with the fastest lap time in the previous race. A rolling average was then created from these ratios for the previous races on the season in order to "reduce the punishment" to drivers who did not perform well in the race occuring right before the one being predicted.
						# '''),
						# html.Br(),
						# html.Div(children = [
						#     dash_table.DataTable(
						#     id = "pr_preds",
						#     columns=[{"name": i, "id": i} for i in pr_predictions_df.columns],
						#     data=pr_predictions_df.to_dict("rows"),
						#     style_table={'maxWidth': '100%'},
						#     style_header={'backgroundColor': 'rgb(30, 30, 30)', 
						#         #'whiteSpace': 'normal',
						#         'height': 'auto',
						#         #'width' : '20px'
						#     },
						#     style_cell={
						#         'backgroundColor': 'rgb(50, 50, 50)',
						#         'color': 'white',
						#         #'width' : '20px'
						#     },
						#     style_cell_conditional=[{
						#         'textAlign': 'center'
						#     }],
						#     style_as_list_view=True,
						#     ),
						# ],
						# style = {'width': '40%', 'margin' : 'auto'}
						# ),
						# html.Br(),
						# dcc.Markdown('''
						#     There is a discrepancy between the two models shown here, and further investigation still needs to be done into the feature engineering model and process to determine how the variables are being weighted to determine race order. Because the model with feature engineered variables uses averages of historical data, though, it could potentially perform better as the season goes on.
						# '''),   
				]),
				dbc.Tab(label="Lap simulation",tab_id="lapsimulation",children= [
					html.Br(),
					html.P('This section examines the effect of key racecar parameters on the performance over a lap. This is implemented through formulating a trajectory optimization of a 3-DOF vehicle model. This optimal control problem (OCP) is transcribed to a nonlinear programming problem (NLP) through OpenMDAO Dymos (open-source). The NLP is solved with the open-source IPOPT solver. A design of experiments (DOE) is constructed with parameters such as the maximum engine power and vehicle lift and drag coefficients. The DOE is evaluated and fed into a radial basis function surrogate model. This model allows for the continous manipulation of each of the design variables.'),
					html.P('A telemetry plot is displayed below, with a choice of which data trace to display. The semi-transparent lines represent all the entries in the DOE. Below that graph, the optimal racing line of the vehicle is shown, colored by the velocity. The user can choose between the absolute velocity, and a velocity relative to a vehicle with mid-range design variables. Currently various tracks are available for evaluation, with more to be added in the future.'),
					html.Br(),
					html.P('Choose the track:'),
					dbc.Row([
					dbc.Col([
						dcc.Dropdown(id='trackselect',options=[{'label':'Bahrain','value':0},{'label':'Bahrain short','value':1},{'label':'Barcelona','value':2},{'label':'Abu Dhabi','value':3},{'label':'Portimao','value':4}],value=4)
						]),
					dbc.Col([])
					]),
					html.Br(),
					html.P('Select design variables:'),
					dbc.Row([
						dbc.Col([
							html.Div(children=[
								html.Div(id='slider1text'),
								dcc.Slider(id='slider1',min=lbounds[0],max=ubounds[0],step=ubounds[0]/100,value=(ubounds[0]+lbounds[0])/2)
							])
						]),
						dbc.Col([
							html.Div(children=[
								html.Div(id='slider2text'),
								dcc.Slider(id='slider2',min=lbounds[1],max=ubounds[1],step=ubounds[1]/100,value=(ubounds[1]+lbounds[1])/2)
							])
						]),
						dbc.Col([
							html.Div(children=[
								html.Div(id='slider3text'),
								dcc.Slider(id='slider3',min=lbounds[2],max=ubounds[2],step=ubounds[2]/100,value=(ubounds[2]+lbounds[2])/2)
							])
						])
					]),
					dbc.Row([
						dbc.Col([
							html.P('Select a data trace to display:'),
							dcc.Dropdown(id='traceradio',options=[{'label':'Velocity','value':0},{'label':'Time','value':1},{'label':'Distance from centerline','value':2}],value=0)
						]),
						dbc.Col([
							html.P('Select relative/absolute data for the track plot:'),
							dcc.Dropdown(id='deltaabs',options=[{'label':'Absolute','value':0},{'label':'Relative','value':1}],value=0)
						])
					]),
					html.Br(),
					html.Div(id='laptimetext'),
					html.Br(),
					html.Div(className="loader-wrapper",children=[
					dcc.Loading(id='traceLoading',color="#fc7303",children=[
					dcc.Graph(className='div-for-charts',id='lapSimGraph')
					])]),
					html.Br(),
					html.Div(className="loader-wrapper",children=[
					dcc.Loading(id='trackLoading',color="#fc7303",children=[
					dcc.Graph(id='trackGraph',style={'width':'100%','height':'70vh','display':'flex','flex-direction':'column'})
					])])
				]),
				dbc.Tab(label = "Contact", tab_id="contact", children = [
						html.Br(),
						dcc.Markdown('''
							# Contact Us
						'''),
						dcc.Markdown('''
							You can email us [here](mailto:oneformulae@gmail.com).
						'''),
						# dbc.Row(
						# [
						# 	dbc.Col(
						# 		dbc.FormGroup(
						# 		[
						# 			dbc.Label("Name", html_for="example-name"),
						# 			dbc.Input(type="text", id="example-name", placeholder="Name"),
						# 		]
						# 		),
						# 	),
						# 	dbc.Col(
						# 		dbc.FormGroup(
						# 		[
						# 			dbc.Label("Email", html_for="example-email"),
						# 			dbc.Input(type="email", id="example-email", placeholder="Email Address"),
						# 		]
						# 		),
						# 	)
						# ]
						# ),
						# html.Div(
						# 	[
						# 		dbc.Label("Comments", html_for="comments"),
						# 		dbc.Textarea(
						# 			id="comments",
						# 			bs_size="lg",
						# 			className="mb-3",
						# 		),
						# 	]
						# ),
						# html.Div(children = [
						# 	dbc.Button("Submit", id='submit-button', color="secondary"),
						# 	html.Span(id="submit_message", style={"vertical-align": "middle"})
						# ]),
						html.Br(),
						dcc.Markdown('''
							# About Us
							[Pieter de Buck](http://pieterdebuck.com) is passionate about anything that has to do with transportation (cars, planes, and rockets) and has always been a huge Formula 1 fan. He is currently at the University of Michigan getting his M.S. in Aerospace Engineering. Prior to that, he received his B.S. in Mechanical Engineering from Carnegie Mellon University, where he was on the Formula SAE team that built and designed a new car every year for competitions in North America and Canada.
						'''),
						html.Br(),
						dcc.Markdown('''
							[Adeline Shin](http://adelineshin.com) is a recent F1 fan, but has always been interested in visualizing and predicting outcomes for her interests and hobbies. She is passionate about healthcare data science, and hopes to use health data to improve future healthcare outcomes. Adeline is currently at Columbia Mailman School of Public Health getting an M.S. in Biostatistics, and was previously at Carnegie Mellon University, where she received her B.S. in Chemical Engineering and Biomedical Engineering. 
						'''),
						html.Br(),
						html.Div(children = [
							html.Img(src=app.get_asset_url('./PieterGrad_0592.jpeg')),
						],
						style={'textAlign': 'center'}
						),
				]),
			],
			id="tabs",
			active_tab="home",
		)
	]
)

################################################################
# App Callback
@app.callback(
	[Output(component_id='my-output', component_property='figure'),Output(component_id='driver1', component_property='options'),Output(component_id='driver2', component_property='options'),Output(component_id='race_name', component_property='options'), Output(component_id='deltaType', component_property='options')],
	[Input(component_id='year', component_property='value'), Input(component_id='race_name', component_property='value'),Input(component_id='filter',component_property='value')]
)
def update_race_graph(year, race_name,filterVal):
	dataContainer.race_table,dataContainer.year_races,dataContainer.color_palette,dataContainer.colored_df = create_race_table(year,race_name)
	dataContainer.filter = filterVal
	drivers = dataContainer.getDriverNames()
	race_names = dataContainer.getRaceNames()
	driver_dict = [{'label': i, 'value': i} for i in drivers]
	race_dict = [{'label': i, 'value': i} for i in race_names]
	delta_dict = copy.deepcopy(driver_dict)
	delta_dict.extend([{'label': 'Median','value': 'median'},{'label': 'First place','value': 'min'},{'label': 'Last place','value': 'max'}])
	return dataContainer.plotRaceGraph(),driver_dict,driver_dict,race_dict,delta_dict

@app.callback(
	Output(component_id='my-output2', component_property='figure'),
	[Input(component_id='driver1', component_property='value'),Input(component_id='driver2', component_property='value'),Input(component_id='year', component_property='value'), Input(component_id='race_name', component_property='value')]
)
def update_comparison_graph(driver1,driver2,year,race_name):
	dataContainer.race_table,dataContainer.year_races,dataContainer.color_palette,dataContainer.colored_df = create_race_table(year,race_name)
	return dataContainer.plotRaceComparisonGraph(driver1,driver2)

@app.callback(
	Output(component_id='deltaGraph', component_property='figure'),
	[Input(component_id='deltaType', component_property='value'),Input(component_id='year', component_property='value'), Input(component_id='race_name', component_property='value')]
)
def update_delta_graph(deltaType,year,race_name):
	dataContainer.race_table,dataContainer.year_races,dataContainer.color_palette,dataContainer.colored_df = create_race_table(year,race_name)
	return dataContainer.plotDeltaGraph(deltaType)

@app.callback(
	[Output(component_id='my-table', component_property='children'), Output(component_id='all_drivers', component_property='options')],
	[Input(component_id='all_drivers', component_property='value')]
)
def update_driver_history(driver_name_1):
	output = []
	dataContainer.driver_history_table = create_driver_table(driver_name_1)
	for i in range(len(dataContainer.driver_history_table.year.unique())): 
		year = dataContainer.driver_history_table.year.unique()[i]
		driver_table_year = dataContainer.driver_history_table[dataContainer.driver_history_table.year==year]
		driver_table_year.columns = ["Year", "Race Name", "Date", "Constructor", "Qualifying Position", "Fastest Qualifying Time", "Final Race Position",
			"Fastest Lap Time (Race)", "Cumulative Season Wins", "Cumulative Season Points", "Driver Standing After Race"]
		driver_table_year = driver_table_year.drop(driver_table_year.columns[0], axis=1)
		driver_table_year['Final Race Position'].fillna('DNF',inplace=True)
		driver_table_year['Qualifying Position'].fillna('DNQ',inplace=True)
		driver_table_year['Fastest Qualifying Time'].fillna('None',inplace=True)
		driver_table_year['Fastest Lap Time (Race)'].fillna('None',inplace=True)
		# for i in range(len(driver_table_year['Final Race Position'])):
		#   print(driver_table_year['Final Race Position'].iloc[i])
		#   if driver_table_year['Final Race Position'].iloc[i] == :
		#       print(i)
		#       driver_table_year['Final Race Position'][i] = 48903
		output.append(
			html.Div(children = [
				html.Details([
					html.Summary(str(year)+' - Final Standing: '+str(driver_table_year['Driver Standing After Race'].iloc[0])+', Wins: '+str(driver_table_year['Cumulative Season Wins'].iloc[0])),
					dash_table.DataTable(id='my-table'+str(i), columns=[{"name": j, "id": j} for j in driver_table_year.columns], 
					data=driver_table_year.to_dict("records"),
					style_header={'backgroundColor': 'rgb(30, 30, 30)', 
						'whiteSpace': 'normal',
						'height': 'auto',
						'max_width' : '100px'
					},
					style_cell={
						'backgroundColor': 'rgb(50, 50, 50)',
						'color': 'white',
						'max_width' : 'auto'
					},
					style_cell_conditional=[{
						'textAlign': 'left'
					}]
					# style_table={'overflowX': 'auto'},
					# fixed_rows={'headers': True}
					),
				],open=None),

			])
		)
	drivers_sorted = driver_history_df.sort_values("driverName")
	driver_names = drivers_sorted.driverName.unique()
	all_drivers = [{'label': i, 'value': i} for i in driver_names]
	return output, all_drivers

@app.callback(
	Output(component_id='qualiFormGraph', component_property='figure'),
	[Input(component_id='chart_switch', component_property='value'),Input(component_id='quali_range',component_property='value')]
)
def update_form_graph(chart_switch,quali_range):
	fig = dataContainer.update_form_graph(chart_switch,quali_range)
	return fig

@app.callback(
	Output(component_id='standings_graph',component_property='figure'),
	[Input(component_id='standings_year', component_property='value')]
)
def update_standings_graph(year):
	return dataContainer.plotStandingsGraph(year)

# @app.callback(
# 	Output(component_id='submit_message', component_property='children'),
# 	[Input(component_id='submit-button', component_property="n_clicks")],
# 	[State(component_id='example-name', component_property='value'), State(component_id='example-email', component_property='value'), State(component_id='comments', component_property='value'),]
# )
# def submit_form(n_clicks, name, email, text):
# 	click_value = n_clicks
# 	if click_value is None:
# 		raise dash.exceptions.PreventUpdate
# 	elif click_value >= 1:
# 		message = Mail(
# 			from_email='adelineshin2015@gmail.com',
# 			to_emails='adelineshin@yahoo.com',
# 			subject='Sending with Twilio SendGrid is Fun',
# 			html_content='<strong>and easy to do anywhere, even with Python</strong>')
# 		try:
# 			sg = SendGridAPIClient(os.environ.get('SENDGRID_API_KEY'))
# 			response = sg.send(message)
# 			print(response.status_code)
# 			print(response.body)
# 			print(response.headers)
# 		except Exception as e:
# 			print(e.message)

# 		# sg = sendgrid.SendGridAPIClient(apikey=os.environ.get('SENDGRID_API_KEY'))
# 		# from_email = Email("adelineshin@yahoo.com")
# 		# subject = "Hello World from the SendGrid Python Library!"
# 		# to_email = Email("adelineshin2015@gmail.com")
# 		# content = Content("text/plain", "Hello, Email!")
# 		# mail = Mail(from_email, subject, to_email, content)
# 		# response = sg.client.mail.send.post(request_body=mail.get())
# 		# print(response.status_code)
# 		# print(response.body)
# 		# print(response.headers)

# 		# mail_handler = MailHandler()
# 		# mail_handler.register_address(
# 		#     address='app181910125@heroku.com',
# 		#     secret='116c54342c2f62c389ca',
# 		#     callback=my_callback_function
# 		# )

# 		# email_body = "New Message from: " + str(email) + str('\n') + str(text)
# 		# me = email
# 		# recipient = '85ccb3a2d43c1c6ad507@cloudmailin.net'
# 		# subject = "New Comment on F1Data.io from " + str(name)

# 		# email_server_host = 'smtp.gmail.com'
# 		# port = 587
# 		# email_username = 'app181910125@heroku.com'
# 		# email_password = 'NrYwmoMYCfDujt9kzr7VJJwF'

# 		# msg = MIMEMultipart('alternative')
# 		# msg['From'] = me
# 		# msg['To'] = recipient
# 		# msg['Subject'] = subject

# 		# msg.attach(MIMEText(email_body, 'html'))

# 		# server = smtplib.SMTP(email_server_host, port)
# 		# server.ehlo()
# 		# server.starttls()
# 		# server.login(email_username, email_password)
# 		# server.sendmail(me, recipient, msg.as_string())
# 		# server.close()
# 		submit_message = str("   Your comment has been submitted. Thank you!")
# 		return submit_message

@app.callback(
	[Output(component_id='lapSimGraph',component_property='figure'),Output(component_id='trackGraph',component_property='figure'),Output(component_id='slider1text',component_property='children'),Output(component_id='slider2text',component_property='children'),Output(component_id='slider3text',component_property='children'),Output(component_id='laptimetext',component_property='children')],
	[Input(component_id='slider1', component_property='value'),Input(component_id='slider2', component_property='value'),Input(component_id='slider3', component_property='value'),Input(component_id='deltaabs',component_property='value'),Input(component_id='traceradio',component_property='value'),Input(component_id='trackselect',component_property='value')]
)
def update_track_graph(val1,val2,val3,absolute,radioval,track):

	if track == 0:
		selectedTrack = tracks.Bahrain
	elif track == 1:
		selectedTrack = tracks.Bahrain_short
	elif track == 2:
		selectedTrack = tracks.Barcelona
	elif track == 3:
		selectedTrack = tracks.AbuDhabi
	elif track == 4:
		selectedTrack = tracks.Portimao

	if selectedTrack != dataContainer.track:
		dataContainer.updateTrack(selectedTrack)
	#str1 = 'Center of pressure (front): '+str(np.round(val1,3))+' m'
	str1 = 'Maximum power: '+str(np.round(val1,3))+' W'
	str2 = 'Lift coefficient (ClA): '+str(np.round(val2,3))
	str3 = 'Drag coefficient (CdA): '+str(np.round(val3,3))
	x = [val1,val2,val3]
	dataContainer.updateRBF(x)

	traceChanged = True
	if dataContainer.prevTrace == radioval:
		traceChanged = False
	dataContainer.prevTrace = radioval

	tracefig = dataContainer.plotTraceGraph(radioval)

	#keys = ['xaxis.range[0]','xaxis.range[1]','yaxis.range[0]','yaxis.range[1]']
	
	#checking if we changed the telemetry trace. If we did we shouldn't take the time to update the track plot
	if traceChanged == False:
		# if relayoutData is not None and keys[0] in relayoutData:
		# 	xaxisrange = [relayoutData[keys[0]],relayoutData[keys[1]]]
		# 	yaxisrange = [relayoutData[keys[2]],relayoutData[keys[3]]]
		# 	dataContainer.trackfig = dataContainer.plotTrackGraph(absolute,xaxisrange,yaxisrange)
		# else:
		dataContainer.trackfig = dataContainer.plotTrackGraph(absolute)

	laptimestring = 'Lap time: '+str(np.round(dataContainer.laptime,3))+' s'

	return [tracefig,dataContainer.trackfig,str1,str2,str3,laptimestring]

@app.callback(
	Output(component_id='tabs',component_property='active_tab'),
	[Input(component_id='url',component_property='pathname')]
)
def changeTab(pathname):
	if pathname == None or pathname == '/':
		return 'home'
	return pathname[1:]

@app.callback(
	Output(component_id='url-output',component_property='pathname'),
	[Input(component_id='tabs',component_property='active_tab')]
)
def changeTab2(tab):
	return '/'+tab



# @app.callback(
# 	Output(component_id='lapSimGraph',component_property='figure'),
# 	[Input(component_id='traceradio',component_property='value'),Input(component_id='slider1', component_property='value'),Input(component_id='slider2', component_property='value'),Input(component_id='slider3', component_property='value')]
# )
# def update_trace_graph(radioval,val1,val2,val3):
# 	return dataContainer.plotTraceGraph(radioval)

# @app.callback(
# 	Output(component_id='trackGraph',component_property='figure'),
# 	[Input('trackGraph', 'relayoutData'),Input(component_id='deltaabs',component_property='value')]
# )
# def display_relayout_data(relayoutData,absolute):
# 	print(relayoutData)
# 	keys = ['xaxis.range[0]','xaxis.range[1]','yaxis.range[0]','yaxis.range[1]']
	
# 	if relayoutData is not None and keys[0] in relayoutData:
# 		xaxisrange = [relayoutData[keys[0]],relayoutData[keys[1]]]
# 		yaxisrange = [relayoutData[keys[2]],relayoutData[keys[3]]]
# 		return dataContainer.plotTrackGraph(absolute,xaxisrange,yaxisrange)
# 	else:
# 		return dataContainer.plot

################################################################
# Load to Dash
if __name__ == "__main__":
	app.run_server(debug=True)