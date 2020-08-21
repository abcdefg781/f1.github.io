import time

import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly import tools as tls
from dash.dependencies import Input, Output, State
import datetime as dt
import copy
import warnings

BS = "https://stackpath.bootstrapcdn.com/bootswatch/4.5.0/lux/bootstrap.min.css"
app = dash.Dash(external_stylesheets = [dbc.themes.BOOTSTRAP])
app.config['suppress_callback_exceptions'] = True
warnings.filterwarnings("ignore")

################################################################
# Import data and Functions
# Import all the data
drivers_df = pd.read_csv("./f1db_csv/drivers.csv").drop(columns = "url")
lap_times_df = pd.read_csv("./f1db_csv/lap_times.csv")
results_df = pd.read_csv("./f1db_csv/results.csv")
constructors_df = pd.read_csv("./f1db_csv/constructors.csv")
races_df = pd.read_csv("./f1db_csv/races.csv")
driver_history_df = pd.read_csv("./f1db_csv/driver_history.csv")
constructor_colors_df = pd.read_csv("./f1db_csv/constructors_colors.csv")
sp_prediction_df = pd.read_csv("./sp_2020_predictions.csv")

# Clean some names and create new variables
# drivers_df
drivers_df["number"] = drivers_df["number"].replace({r"\N": None})
drivers_df["driverName"] = drivers_df["forename"].str.cat(drivers_df["surname"],sep = " ")
drivers_df = drivers_df.drop(columns = ["forename", "surname"])

# lap_times_df
clean_lt_df = lap_times_df[["raceId", "driverId", "lap", "milliseconds"]]
clean_lt_df["seconds"] = clean_lt_df.milliseconds / 1000
clean_lt_df = clean_lt_df.drop(columns = "milliseconds")

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
#quali_form_df = driver_history_df.merge(minlap_df,on='raceId')
driver_history_df.drop(columns='raceId',inplace=True)

def create_race_table(year, race_name):
	races_temp = races_df[races_df.year == year]
	race_id = int(races_temp.raceId[races_temp.name == race_name])
	lap_times_1 = clean_lt_df[clean_lt_df.raceId == race_id]
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
		self.race_table,self.year_races,self.color_palette,self.colored_df = create_race_table(2020, "British Grand Prix")
		self.driver_history_table = create_driver_table("Lewis Hamilton")
		self.driver_yr_history_table = create_driver_table("Lewis Hamilton")

	def plotRaceGraph(self):
		#fig = px.line(self.race_table, x = "lap", y = "seconds", color = "driverName", hover_name = "driverName", hover_data = {"driverName" : False, "constructorRef" : True}, 
			#color_discrete_map = self.color_palette
			#)
		fig = go.Figure()
		# df_grouped = [y for x,y in self.race_table.groupby('driverName',as_index=False)]
		# for i in range(len(df_grouped)):
		# 	driverName = df_grouped[i]['driverName'].iloc[0]
		# 	fig.add_trace(go.Scatter(x = df_grouped[i]['lap'], y = df_grouped[i]['seconds'],mode='lines',name=driverName))

		df_team_grouped = [y for x,y in self.race_table.groupby('constructorRef',as_index=False)]

		for i in range(len(df_team_grouped)):
			df_grouped = [y for x,y in df_team_grouped[i].groupby('driverName',as_index=False)]
			for j in range(len(df_grouped)):
				df_driver = df_grouped[j]
				name = df_driver['driverName'].iloc[0]
				color = self.colored_df[self.colored_df.driverName==name]['color'].iloc[0]
				if j == 0:
					line = go.Scatter(x=df_driver['lap'],y=df_driver['seconds'],name=name,mode='lines',line=go.scatter.Line(color=color))
				else:
					line = go.Scatter(x=df_driver['lap'],y=df_driver['seconds'],name=name,mode='lines',line=go.scatter.Line(color=color,dash='dot'))
				fig.add_trace(line)

		#fig.update_layout(legend_title_text=None)

		fig.update_layout(plot_bgcolor="#323130",
			paper_bgcolor="#323130",font=dict(color="white"))

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

		fig = go.Figure()
		fig.add_trace(go.Scatter(x = df_3['lap'], y = df_3['delta'],mode='lines',name='Lap Delta',customdata=df_3['hovertext2'],hovertemplate=
			'<br>%{customdata}'))
		fig.add_trace(go.Scatter(x = df_3['lap'], y = df_3['cumdelta'],mode='lines',name='Cumulative Delta',customdata=df_3['hovertext'],hovertemplate=
			'<br>%{customdata}'))


		fig.update_layout(
			title='Gap from '+driver1+' to '+driver2,
			xaxis_title="Lap",
			yaxis_title="Delta (s)",
			hovermode="x unified"
			)

		fig.update_layout(plot_bgcolor="#323130",
			paper_bgcolor="#323130",font=dict(color="white"))

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
		return fig

	def update_form_graph(self,chart_switch):
		df_5 = copy.deepcopy(quali_form_df)
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
				yaxis_title="Qualifying lap time ratio"
				)
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

#dataContainer object creation
dataContainer = dataContainer()

################################################################
# App Layout
app.layout = dbc.Container(
	[
		dcc.Store(id="store"),
		html.Br(),
		html.Br(),
		html.H1("F1Data.io"),
		dbc.Tabs(
			[
				dbc.Tab(label="Home", tab_id="home", children = [
						html.Br(),
						html.P("Welcome to F1Data.io. The graph below shows the data from the latest race and can be changed to any race."),
					]),
				dbc.Tab(label="Driver Comparison", tab_id="driver_comp", children = [
						html.Br(),
						dbc.Row([
							dbc.Col(
								dcc.Dropdown(id='year',value=2020,clearable=False,searchable=False,options=[{'label': i, 'value': i} for i in races_df['year'].unique()])
							),
							dbc.Col(
								dcc.Dropdown(id='race_name',value='Spanish Grand Prix',clearable=False,searchable=False)
							)
						]),    
						html.Br(),
						dcc.Graph(id='my-output'),
						html.Br(),
						html.P("Select two drivers to view their relative lap times for any given race."),
						dbc.Row([
							dbc.Col(
								dcc.Dropdown(id='driver1',value='Lewis Hamilton',clearable=False,searchable=False)
							),
							dbc.Col(
								dcc.Dropdown(id='driver2',value='Valtteri Bottas',clearable=False,searchable=False)
							)
						]),
						html.Br(),
						dcc.Graph(id='my-output2'),
						html.Br(),
						html.P("Select a driver as the reference or median, min, or max time."),
						dcc.Dropdown(id='deltaType',value='min',clearable=False,searchable=False),
						html.Br(),
						dcc.Graph(id='deltaGraph')
					]),
				dbc.Tab(label = "Driver History", tab_id="collapses", children = [
						html.Br(),
						html.P("Select one driver to view their history. It is possible to search the dropdown menu"),
						dcc.Dropdown(id='all_drivers', clearable=False,searchable=True,value='Lewis Hamilton'),
						html.Br(),
						html.Div(id='my-table')
				]),
				dbc.Tab(label = "Qualifying Form", tab_id="qualitab", children = [
						html.Br(),
						html.P("Select chart style."),
						dcc.Dropdown(id='chart_switch', clearable=False,searchable=False,value=0,options=[{'label':'Linear fit','value':0},{'label':'Quadratic fit','value':1},{'label':'Raw data','value':2}]),
						html.Br(),
						dcc.Graph(id='qualiFormGraph')
				]),
				dbc.Tab(label = "Race Predictions", tab_id="predictions", children =[
						html.Br(),
						dcc.Markdown('''
							**Predictions for the 2020 Spanish Grand Prix**  
							The predictive model is updated after qualifying every week. The model uses an XGBoost algorithm to predict the total race time for each driver, which is then sorted to obtain predicted driver position for the race. The model is dependent on the weather on the day of the race, constructors (as well as constructor standing and points), qualifying results, and results from the previous races of the season.
						'''),
						html.Br(),
						dcc.Markdown('''

						''')
				]),
			],
			id="tabs",
			active_tab="home",
		),
		html.Div(id="tab-content", className="p-4"),
	]
)

################################################################
# App Callback
@app.callback(
	[Output(component_id='my-output', component_property='figure'),Output(component_id='driver1', component_property='options'),Output(component_id='driver2', component_property='options'),Output(component_id='race_name', component_property='options'), Output(component_id='deltaType', component_property='options')],
	[Input(component_id='year', component_property='value'), Input(component_id='race_name', component_property='value')]
)
def update_race_graph(year, race_name):
	dataContainer.race_table,dataContainer.year_races,dataContainer.color_palette,dataContainer.colored_df = create_race_table(year,race_name)
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
		# 	print(driver_table_year['Final Race Position'].iloc[i])
		# 	if driver_table_year['Final Race Position'].iloc[i] == :
		# 		print(i)
		# 		driver_table_year['Final Race Position'][i] = 48903
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

	driver_names = driver_history_df.driverName.unique()
	all_drivers = [{'label': i, 'value': i} for i in driver_names]
	return output, all_drivers

@app.callback(
	Output(component_id='qualiFormGraph', component_property='figure'),
	[Input(component_id='chart_switch', component_property='value')]
)
def update_form_graph(chart_switch):
	return dataContainer.update_form_graph(chart_switch)

################################################################
# Load to Dash
if __name__ == "__main__":
	app.run_server(debug=True)