# Multi-Linear Regression
This is a practice using multi-linear regression on a motorcycle data set accumulated from Bikez.com, predicting engine horsepower given other metrics of motocycle details.

**Files**
- MotorcycleRegression.py - Main driving file used for regression task
- RegressionLib.py - personally coded linear regression file to compare to sklearn regression
- all_bikez_curated.csv - data file for program, contains motorcycle info

##Inital look
Began by checking the distribuition of displacements in the selected dataset.

![Motorcycle CC distribution](https://github.com/MatthewPeplinski/Motorcycle-Regression/blob/main/total_distribution.png)

## Data processing
I decided to restrict the data to above 130 cc's which is in line with Wisconsins motor vehicle laws to determine what is legally considered a motorcycle and then again to above 250 cc's based on my own knowledge of relevant displacements to the problem I wanted to solve.
![Motorcycle cc distribution over 250 cc's](https://github.com/MatthewPeplinski/Motorcycle-Regression/blob/main/over_250_cc_dist.png)

## Final results
Also made an effort to make my own linear regression in order to compare to SKLearn's regression library. The results are printed to terminal as shown below:

Sklearn Linear
R^2: 0.7304619192698891
RMSE: 23.956414185100616
--------------------
Personal Linear
R^2: 0.7304619192698891
RMSE: 23.956414185100616
--------------------
Sklearn Linear with higher base CC
R^2: 0.8086845346245763
RMSE: 25.60126208815181
