import pandas as pd
import numpy as np
from scipy import stats
import scipy.stats
from scipy.stats import norm
from math import log, floor
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

head = list(pd.DataFrame(data=pd.read_csv("data_bee_types/LS_spatial_D_x.csv")))

#test_x = [60, 50, 40, 30, 20, 10, 0, 10, 20, 30, 40, 50, 60, 50, 40, 30, 20, 10, 0, 10, 20, 30, 40, 50, 60, 50, 40, 30, 20, 10, 0, 10, 20, 30, 40, 50, 60]
#test_y = [30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30]
#head = ["BT05A-2"]

""" CONSTANTS """
#-------------------------------------------------------------------------------
# D_r = 400
# a = 0.4
# D_r = 1000
# a = 1
#-------------------------------------------------------------------------------

"""-begin------remove empty entires------------------------------------------"""
def X_remove_NaN(item):
    X_dataset_entropy = pd.DataFrame(data=pd.read_csv("data_bee_types/LS_spatial_D_x.csv"), columns=[item])
    X_dataset_wo_NAN = []
    for m in range(len(X_dataset_entropy)):
        check = X_dataset_entropy.iloc[m].item()
        if str(check) != "nan":
            X_dataset_wo_NAN.append(check)
    return X_dataset_wo_NAN

def Y_remove_NaN(item):
    Y_dataset_entropy = pd.DataFrame(data=pd.read_csv("data_bee_types/LS_spatial_D_y.csv"), columns=[item])
    Y_dataset_wo_NAN = []
    for m in range(len(Y_dataset_entropy)):
        check = Y_dataset_entropy.iloc[m].item()
        if str(check) != "nan":
            Y_dataset_wo_NAN.append(check)
    return Y_dataset_wo_NAN

def T_remove_NaN(item):
    T_dataset = pd.DataFrame(data=pd.read_csv("data_bee_types/temperature_runs.csv"), columns=[item])
    T_dataset_wo_NAN = []
    for m in range(len(T_dataset)):
        check = T_dataset.iloc[m].item()
        if str(check) != "nan":
            T_dataset_wo_NAN.append(check)
    return T_dataset_wo_NAN
"""-end--------remove empty entires------------------------------------------"""

"""-begin--------calculate probability for spending time at a temperature-----------------"""
def probability_T(item):
    T_dataset_wo_NAN = T_remove_NaN(item)
    T_minmax_data = pd.DataFrame(data=pd.read_csv("data_bee_types/temperature_variables.csv"), columns=[item])
    T_ideal = T_minmax_data.iloc[1].item()
    prob_dataset = []
    for T in range(len(T_dataset_wo_NAN)):
        prob_dataset.append(np.abs(T_dataset_wo_NAN[T] - T_ideal))
    return prob_dataset
"""-end--------calculate probability for spending time at a temperature-------------------"""

"""-begin------calculate duration walking-----------------"""
def calc_dur_walk(item):
    list_vel = []
    X_dataset_wo_NAN = X_remove_NaN(item) #test_x
    Y_dataset_wo_NAN = Y_remove_NaN(item) #test_y
    n = len(X_dataset_wo_NAN) - 1
    for t in range(n):
        velocity = np.sqrt((X_dataset_wo_NAN[t + 1] - X_dataset_wo_NAN[t]) ** 2 + (Y_dataset_wo_NAN[t + 1] - Y_dataset_wo_NAN[t]) ** 2)
        list_vel.append(velocity)

    T_dataset_wo_NAN = T_remove_NaN(item)
    T_minmax_data = pd.DataFrame(data=pd.read_csv("data_bee_types/temperature_variables.csv"), columns=[item])
    T_ideal = T_minmax_data.iloc[1].item()

    walk_vel = 0.4
    duration_walk = 0
    all_walk_durations = []
    walk_temperatures = []
    mean_delta_walk_temp = []
    for v in range(len(list_vel[:-1])):
        if list_vel[v] >= walk_vel:
            duration_walk += 1
            walk_temperatures.append(T_dataset_wo_NAN[v])
            if list_vel[v+1] < walk_vel:
                all_walk_durations.append(duration_walk)
                mean_delta_walk_temp.append(np.abs(np.mean(walk_temperatures) - T_ideal))
                duration_walk = 0
                walk_temperatures = []
            elif v == len(list_vel[:-2]):
                all_walk_durations.append(duration_walk)
                mean_delta_walk_temp.append(np.abs(np.mean(walk_temperatures) - T_ideal))

    return all_walk_durations, mean_delta_walk_temp
#print(calc_vel(testbee))
#plt.show()
"""-end--------calculate duration walking-----------------"""

"""-begin------calculate duration stopping-----------------"""
def calc_dur_stop(item):
    list_vel = []
    X_dataset_wo_NAN = X_remove_NaN(item) #test_x
    Y_dataset_wo_NAN = Y_remove_NaN(item) #test_y
    n = len(X_dataset_wo_NAN) - 1
    for t in range(n):
        velocity = np.sqrt((X_dataset_wo_NAN[t + 1] - X_dataset_wo_NAN[t]) ** 2 + (Y_dataset_wo_NAN[t + 1] - Y_dataset_wo_NAN[t]) ** 2)
        list_vel.append(velocity)

    T_dataset_wo_NAN = T_remove_NaN(item)
    T_minmax_data = pd.DataFrame(data=pd.read_csv("data_bee_types/temperature_variables.csv"), columns=[item])
    T_ideal = T_minmax_data.iloc[1].item()

    stop_vel = 0.4
    duration_stop = 0
    all_stop_durations = []
    stop_temperatures = []
    mean_delta_stop_temp = []
    for v in range(len(list_vel[:-1])):
        if list_vel[v] <= stop_vel:
            duration_stop += 1
            stop_temperatures.append(T_dataset_wo_NAN[v])
            if list_vel[v+1] > stop_vel:
                all_stop_durations.append(duration_stop)
                mean_delta_stop_temp.append(np.abs(np.mean(stop_temperatures) - T_ideal))
                duration_walk = 0
                walk_temperatures = []
            elif v == len(list_vel[:-2]):
                all_stop_durations.append(duration_stop)
                mean_delta_stop_temp.append(np.abs(np.mean(stop_temperatures) - T_ideal))
    return all_stop_durations, mean_delta_stop_temp
#print(calc_vel(testbee))
#plt.show()
"""-end--------calculate duration stopping-----------------"""

def myCurve(x,lam,d):
    return lam * np.exp(-lam*x+d)

# for testbee in head:
#     print()
ALL_WALKS_at_dT = []
ALL_TEMPS_at_walk = []
ALL_STOPS_at_dT = []
ALL_TEMPS_at_stop = []
for testbee in head:
    walking_times, mean_walk_temps = calc_dur_walk(testbee)
    stopping_times, mean_stop_temps = calc_dur_stop(testbee)
    ALL_WALKS_at_dT.extend(walking_times)
    ALL_TEMPS_at_walk.extend(mean_walk_temps)
    ALL_STOPS_at_dT.extend(stopping_times)
    ALL_TEMPS_at_stop.extend(mean_stop_temps)
#plt.hist(ALL_WALKS_at_dT, bins=100)
fig, (subfigure_stop, subfigure_walk) = plt.subplots(1, 2, figsize=(10,5))
fig.set_tight_layout(True)
subfigure_stop.scatter(ALL_TEMPS_at_stop, ALL_STOPS_at_dT, color='black', alpha=0.1)
subfigure_walk.scatter(ALL_TEMPS_at_walk, ALL_WALKS_at_dT, color='black', alpha=0.1)
stop, stop_cov = np.polyfit(ALL_TEMPS_at_stop, ALL_STOPS_at_dT, 1, cov=True)
walk, walk_cov = np.polyfit(ALL_TEMPS_at_walk, ALL_WALKS_at_dT, 1, cov=True)

popt_stop, pcov_stop = curve_fit(myCurve, ALL_TEMPS_at_stop, ALL_STOPS_at_dT, maxfev = 200000, p0=(1,1))
popt_walk, pcov_walk = curve_fit(myCurve, ALL_TEMPS_at_walk, ALL_WALKS_at_dT, maxfev = 200000, p0=(1,1))


print(np.corrcoef(ALL_TEMPS_at_walk,ALL_WALKS_at_dT))
print(np.corrcoef(ALL_TEMPS_at_stop,ALL_STOPS_at_dT))
x = np.linspace(0,10)
print(stop, stop_cov)
print(walk, walk_cov)

subfigure_stop.plot(x, myCurve(x, *popt_stop), linewidth=2, color='red', linestyle='dotted')#, linewidth=1)
subfigure_walk.plot(x, myCurve(x, *popt_walk), linewidth=2, color='blue', linestyle='dotted')#, linewidth=1)
#subfigure_stop.plot(x, np.polyval(stop,ALL_TEMPS_at_stop), color='firebrick', linewidth=1, linestyle='dashed')
#subfigure_walk.plot(x, np.polyval(walk,ALL_TEMPS_at_stop), color='cornflowerblue', linewidth=1, linestyle='dashed')
subfigure_stop.plot(x, stop[0]*x+stop[1], color='red', linewidth=1.1, linestyle='dashed')
subfigure_walk.plot(x, walk[0]*x+walk[1], color='blue', linewidth=1.1, linestyle='dashed')
subfigure_stop.set_xlabel(r'$\Delta T \,[°C]$')
subfigure_stop.set_ylabel("stop duration " r'$t_{s} [s]$')
subfigure_walk.set_xlabel(r'$\Delta T \,[°C]$')
subfigure_walk.set_ylabel("walk duration " r'$t_{w} [s]$')
#plt.xscale('log')
#plt.yscale('log')
plt.show()
