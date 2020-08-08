import pandas as pd

# Import all the data
drivers_df = pd.read_csv("./f1db_csv/drivers.csv").drop(columns = "url")
lap_times_df = pd.read_csv("./f1db_csv/lap_times.csv")
results_df = pd.read_csv("./f1db_csv/results.csv")
constructors_df = pd.read_csv("./f1db_csv/constructors.csv")
races_df = pd.read_csv("./f1db_csv/races.csv")

# Clean some names and create new variables
# drivers_df
drivers_df["number"] = drivers_df["number"].replace({r"\N": None})
drivers_df["driverName"] = drivers_df["forename"].str.cat(drivers_df["surname"],sep = " ")
drivers_df = drivers_df.drop(columns = ["forename", "surname"])

# Create the table of drivers and teams for each race
driver_race_df_1 = pd.merge(results_df[["raceId", "driverId", "constructorId"]], drivers_df[["driverName", "driverRef", "driverId"]], on = ["driverId"])
driver_race_df_2 = pd.merge(driver_race_df_1, constructors_df[["constructorRef", "constructorId", "name"]], on = "constructorId")
driver_race_df_2 = driver_race_df_2.rename({'name': 'constructorName'}, axis=1)
driver_race_df = pd.merge(driver_race_df_2, races_df[["year", "name", "raceId"]], on = "raceId")
driver_race_df = driver_race_df.sort_values(by = "year", ascending = False)

# Add color to every driver for each race
driver_race_df["color"] = [0] * len(driver_race_df.raceId)
for i in range(len(driver_race_df.raceId)):
    if driver_race_df.year[i] == 2020:
        if driver_race_df.constructorName[i] == "Mercedes":
            driver_race_df.loc[i, 9] = "#00D2BE"
            i += 1
        elif driver_race_df.constructorName[i] == "Red Bull":
            driver_race_df.loc[i, 9] = "#0600EF"
            i += 1
        elif driver_race_df.constructorName[i] == "Ferrari":
            driver_race_df.loc[i, 9] = "#C00000"
            i += 1 
        elif driver_race_df.constructorName[i] == "Renault":
            driver_race_df.loc[i, 9] = "#FFF500"
            i += 1 
        elif driver_race_df.constructorName[i] == "Haas":
            driver_race_df.loc[i, 9] = "#787878"
            i += 1     
        elif driver_race_df.constructorName[i] == "Racing Point":
            driver_race_df.loc[i, 9] = "#F596C8"
            i += 1 
        elif driver_race_df.constructorName[i] == "Alpha Tauri":
            driver_race_df.loc[i, 9] = "#C8C8C8"
            i += 1 
        elif driver_race_df.constructorName[i] == "Mclaren":
            driver_race_df.loc[i, 9] = "#FF8700"
            i += 1 
        elif driver_race_df.constructorName[i] == "Alfa Romeo":
            driver_race_df.loc[i, 9] = "#960000"
            i += 1 
        elif driver_race_df.constructorName[i] == "Williams":
            driver_race_df.loc[i, 9] = "#0082FA"
            i += 1
    # elif driver_race_df.year[i] == 2019:
    #     if driver_race_df.constructorName[i] == "Mercedes":
    #         driver_race_df.color[i] = "#00D2BE"
    #         i += 1
    #     elif driver_race_df.constructorName[i] == "Red Bull":
    #         driver_race_df.color[i] = "#1E41FF"
    #         i += 1
    #     elif driver_race_df.constructorName[i] == "Ferrari":
    #         driver_race_df.color[i] = "#DC0000"
    #         i += 1 
    #     elif driver_race_df.constructorName[i] == "Renault":
    #         driver_race_df.color[i] = "#FFF500"
    #         i += 1 
    #     elif driver_race_df.constructorName[i] == "Haas":
    #         driver_race_df.color[i] = "#F0D787"
    #         i += 1     
    #     elif driver_race_df.constructorName[i] == "Racing Point":
    #         driver_race_df.color[i] = "#F596C8"
    #         i += 1 
    #     elif driver_race_df.constructorName[i] == "Toro Rosso":
    #         driver_race_df.color[i] = "#469BFF"
    #         i += 1 
    #     elif driver_race_df.constructorName[i] == "Mclaren":
    #         driver_race_df.color[i] = "#FF8700"
    #         i += 1 
    #     elif driver_race_df.constructorName[i] == "Alfa Romeo":
    #         driver_race_df.color[i] = "#9B0000"
    #         i += 1 
    #     elif driver_race_df.constructorName[i] == "Williams":
    #         driver_race_df.color[i] = "#FFFFFF"
    #         i += 1

print(driver_race_df)
