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

for testbee in head:
    vel = calc_vel(testbee)
    T = T_remove_NaN(testbee)
    plt.scatter(vel, T)

    k, d = np.polyfit(vel, T, 1)
    vel_range = np.linspace(0, max(vel))
    plt.plot(vel_range, k*vel_range+d, linestyle='--', color='red')

    plt.xlabel('velocity[cm/s]')
    plt.ylabel('Temperature[°C]')
    plt.title(testbee)
    plt.savefig("FIG_vel_vs_T/" + str(testbee))

    plt.close()
    #plt.show()
    # plt.scatter(calc_vel(testbee), calc_delta_T(testbee))
    # plt.show()
