import pandas as pd
import numpy as np
from scipy import stats
import scipy.stats
from math import log, floor
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.mlab as mlab
from scipy.optimize import curve_fit

head = list(pd.DataFrame(data=pd.read_csv("data_bee_types/LS_spatial_D_x.csv")))

#testbee = "BT01A-1"
#testbee = "BT06B-1"

control_30_30 = ["BT17A-1", "BT17A-2", "BT17A-3", "BT17A-4", "BT17B-1", "BT17B-2", "BT17B-3", "BT17B-4"]
control_36_36 = ["BT07A-1", "BT07A-2", "BT07A-3", "BT07A-4", "BT07B-1", "BT07B-2", "BT07B-3", "BT07B-4"]
control_ALL = control_30_30 + control_36_36
for item in control_ALL:
    head.remove(item)

bad_bees = ["BT11A-1","BT11A-4", "BT01A-1","BT02B-1", "BT03A-1", "BT03A-2", "BT03A-3", "BT03A-4", "BT03B-1", "BT03B-2", "BT03B-3", "BT03B-4"]
# #BT11A-1 and "BT11A-4 needs to be removed from the bad bees list.
for item in bad_bees:
    head.remove(item)


""" CONSTANTS """
#--------------------------------------------------------------------------------


#--------------------------------------------------------------------------------

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
"""-end--------remove empty entires------------------------------------------"""


"""-begin------calculate velocity for each time step-----------------"""
def calc_vel(item):
    list_vel = []
    X_dataset_wo_NAN = X_remove_NaN(item) #test_x
    Y_dataset_wo_NAN = Y_remove_NaN(item) #test_y
    n = len(X_dataset_wo_NAN) - 1
    timeline = np.linspace(0, n, n)
    for t in range(n):
        velocity = np.sqrt((X_dataset_wo_NAN[t + 1] - X_dataset_wo_NAN[t]) ** 2 + (Y_dataset_wo_NAN[t + 1] - Y_dataset_wo_NAN[t]) ** 2)
        list_vel.append(velocity)
    return list_vel
#plt.plot(calc_theta(testbee))
"""-end--------calculate velocity in for each time step-----------------"""


"""-begin------remove empty entires------------------------------------------"""
def T_remove_NaN(item):
    T_dataset = pd.DataFrame(data=pd.read_csv("data_bee_types/temperature_runs.csv"), columns=[item])
    T_dataset_wo_NAN = []
    for m in range(1,len(T_dataset)):
        check = T_dataset.iloc[m].item()
        if str(check) != "nan":
            T_dataset_wo_NAN.append(check)
    return T_dataset_wo_NAN
"""-end--------remove empty entires------------------------------------------"""

"""-begin------calculate inclination-----------------------------------------"""
def calc_delta_T(item):
    calc_dataset = T_remove_NaN(item)
    delta_T = []
    for i in range(len(calc_dataset) - 1):
        delta_T.append(calc_dataset[i + 1] - calc_dataset[i])
    return delta_T
"""-end--------calculate inclination-----------------------------------------"""
k_list = []
d_list = []
for testbee in head:
    vel = calc_vel(testbee)
    T = T_remove_NaN(testbee)

    plt.scatter(T, vel, color='black',  alpha=0.01)
    k, d = np.polyfit(T, vel, 1)
    k_list.append(k)
    d_list.append(d)
    T_range = np.linspace(min(T), max(T))
    #plt.plot(T_range, T_range * k + d, linestyle='--', color='red', alpha=0.3)

    #plt.xscale('log')
    #plt.yscale('log')
    plt.xlabel('Temperature[°C]')
    plt.ylabel('velocity[cm/s]')
    #plt.title(testbee)
    #plt.savefig("FIG_vel_vs_T/" + str(testbee))
    #plt.close()
mean_k = np.mean(k_list)
mean_d = np.mean(d_list)
T_range_2 = np.linspace(26, 36)
plt.plot(T_range_2, T_range_2 * mean_k + mean_d, linestyle='--', color='red', alpha=1)#0.3)
plt.show()
    # plt.scatter(calc_vel(testbee), calc_delta_T(testbee))
    # plt.show()
