"""
This program uses linear regression to model motorcycle horsepower using sklearn's linear regression
and a personally coded linear regression

https://www.kaggle.com/datasets/emmanuelfwerr/motorcycle-technical-specifications-19702022
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import RegressionLib as rl


data = pd.read_csv("all_bikez_curated.csv", low_memory=False)
#print(data.info())

##I see that there are 10 numeric data catagories in this dataset and >38000 entries
"""
Next will be selecting which catagories I will look at
(#+ denotes a feature that makes sense to use, #-denotes a non-useful feature)

## Power (HP): Variable to be modeled, the measure of work which can be produced by the engine
#- Rating: Bad variable, data source only requires 3 users to enter a rating for it to be shown, causing very small sample
         sizes even on popular bikes. There is no profesional reveiw score for this source
#-  Year: likely correlated to HP since motorcycles have gotten better over the years, but is likely colinear to other metrics
#+ Displacement (ccm): Good variable to use, describes the area inside the engine that the combustion occures in and how much
                       it moved the pistons
#+ Torque (Nm): Good variable option. Measures the amount of rotational power delivered to the back wheel
#- Bore (mm): With Stroke length and engine type, this could be good but alone this metric is questionably useful
#- Fuel Capacity: Bad option, fuel capacity is all over the place on motorcycles
#+ Dry Weight (kg): Good Variable to use, though may be skewed by mopeds on the lower weights
#- Wheelbase (mm): May be interesting, though since data set is using mopeds and motorcycles, the data may be misleading
#- Seat Height (mm): Bad option, basically dictates rider comfort
"""

data_selected = data[["Power (hp)", "Displacement (ccm)", "Torque (Nm)", "Dry weight (kg)"]]
data2 = data_selected.dropna()

#renaming into more usable names
data2 = data2.rename(columns = {"Power (hp)":"Power"})
data2 = data2.rename(columns = {"Displacement (ccm)": "Displacement"})
data2 = data2.rename(columns = {"Torque (Nm)":"Torque"})
data2 = data2.rename(columns = {"Dry weight (kg)": "DryWeight"})

#check changes made to the data
data2.info()

"""
we went from 38298 entries down to 10265 entries. Next is to remove small displacement vehicles 
to narrow the displacement category since I want information on the effects of the variables on
motorcycles and not on scooters. The legal cutoff for scooters is 130 cc in WI so that is where
I will draw the lower bound
"""
data2.Displacement.plot.hist()

"""
I see there are some outlier motorcycels that are harming the data visualization
but after taking them away from the model, it performed worse in R^2. implying the 
larger displacement motorcycles have significant impact
"""
data2.query("Displacement < 3000 and Displacement >= 130").Displacement.plot.hist()

#This is creating a more subjective criteria for the displacement level I want to see
reg_data = data2.query("Displacement < 3000 and Displacement >= 250")
reg_data.info()


X = reg_data[["Displacement", "Torque", "DryWeight"]]
y = reg_data["Power"]

reg = LinearRegression().fit(X, y)
print()
print("Sklearn Linear")
print(f"R^2: {reg.score(X,y)}")
print(f"RMSE: {np.sqrt(np.average((y-reg.predict(X))**2))}")
print("-"*20)

reg_personal = rl.LinearRegression().fit(X,y)
print("Personal Linear")
print(f"R^2: {reg_personal.score(X,y)}")
print(f"RMSE: {np.sqrt(np.average((y-reg_personal.predict(X))**2))}")
print("-"*20)

#this is a look at the data with higher base displacement to see if it works better
X2 = data2[["Displacement", "Torque", "DryWeight"]]
y2 = data2["Power"]
print("Sklearn Linear with higher base CC")
print(f"R^2: {reg.score(X2,y2)}")
print(f"RMSE: {np.sqrt(np.average((y2-reg.predict(X2))**2))}")




