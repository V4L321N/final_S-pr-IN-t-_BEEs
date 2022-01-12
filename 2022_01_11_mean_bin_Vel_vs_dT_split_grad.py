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

NARROW = ["BT01A-1","BT01A-2","BT01A-3","BT01A-4","BT01C-1","BT01C-2","BT01C-3","BT01C-4","BT02A-1","BT02A-2","BT02A-3","BT02A-4","BT02B-1","BT02B-2","BT02B-3","BT02B-4","BT03A-1","BT03A-2","BT03A-3","BT03A-4","BT03B-1","BT03B-2","BT03B-3","BT03B-4","BT06A-1","BT06A-2","BT06A-3","BT06A-4","BT06B-1","BT06B-2","BT06B-3","BT06B-4","BT18A-1","BT18A-2","BT18A-3","BT18A-4","BT18B-1","BT18B-2","BT18B-3","BT18B-4"]
STEEP = ["BT04A-1","BT04A-2","BT04A-3","BT04A-4","BT04B-1","BT04B-2","BT04B-3","BT04B-4","BT05A-1","BT05A-2","BT05A-3","BT05A-4","BT05B-1","BT05B-2","BT05B-3","BT05B-4","BT08A-1","BT08A-2","BT08A-3","BT08A-4","BT08B-1","BT08B-2","BT08B-3","BT08B-4","BT09A-1","BT09A-2","BT09A-3","BT09A-4","BT09B-1","BT09B-2","BT09B-3","BT09B-4","BT10A-1","BT10A-2","BT10A-3","BT10A-4","BT10B-1","BT10B-2","BT10B-3","BT10B-4"]
STEEPEST = ["BT11A-1","BT11A-2","BT11A-3","BT11A-4","BT11B-1","BT11B-2","BT11B-3","BT11B-4","BT12A-1","BT12A-2","BT12A-3","BT12A-4","BT12B-1","BT12B-2","BT12B-3","BT12B-4","BT13A-1","BT13A-2","BT13A-3","BT13A-4","BT13B-1","BT13B-2","BT13B-3","BT13B-4","BT14A-1","BT14A-2","BT14A-3","BT14A-4","BT14B-1","BT14B-2","BT14B-3","BT14B-4","BT15A-1","BT15A-2","BT15A-3","BT15A-4","BT15B-1","BT15B-2","BT15B-3","BT15B-4"]

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

"""----all----"""

T_master_bin = np.arange(27, 37, 0.1)
v_mean_list = np.zeros(100)
m_list = np.zeros(100)

for testbee in head:
    vel = calc_vel(testbee)
    T = T_remove_NaN(testbee)

    T_bin = np.arange(min(T), max(T), 0.1)
    bin_vel_LIST = []
    for i in T_bin:
        bin_vel = 0
        n = 0.000001
        for ii in range(len(vel)):
            if np.abs(T[ii] - i) < 0.1:
                bin_vel += vel[ii]
                n += 1
        bin_vel_LIST.append(bin_vel/n)
    for j in range(len(T_master_bin)):
        for jj in range(len(T_bin)):
            if np.abs(T_master_bin[j]-T_bin[jj]) < 0.1:
                v_mean_list[j] += bin_vel_LIST[jj]
                m_list[j] += 1
    # print(v_mean_list)
    # print(m_list)
    #plt.plot(T_bin, bin_vel_LIST,alpha=0.3)
for v in range(len(v_mean_list)):
    v_mean_list[v] /= m_list[v]
plt.plot(T_master_bin, v_mean_list, label="all")
#plt.show()

"""--------"""

T_master_bin = np.arange(27, 37, 0.1)
v_mean_list = np.zeros(100)
m_list = np.zeros(100)

for testbee in NARROW:
    vel = calc_vel(testbee)
    T = T_remove_NaN(testbee)

    T_bin = np.arange(min(T), max(T), 0.1)
    bin_vel_LIST = []
    for i in T_bin:
        bin_vel = 0
        n = 0.000001
        for ii in range(len(vel)):
            if np.abs(T[ii] - i) < 0.1:
                bin_vel += vel[ii]
                n += 1
        bin_vel_LIST.append(bin_vel/n)
    for j in range(len(T_master_bin)):
        for jj in range(len(T_bin)):
            if np.abs(T_master_bin[j]-T_bin[jj]) < 0.1:
                v_mean_list[j] += bin_vel_LIST[jj]
                m_list[j] += 1
    # print(v_mean_list)
    # print(m_list)
    #plt.plot(T_bin, bin_vel_LIST,alpha=0.3)
for v in range(len(v_mean_list)):
    v_mean_list[v] /= m_list[v]
plt.plot(T_master_bin, v_mean_list, label="narrow")

T_master_bin = np.arange(27, 37, 0.1)
v_mean_list = np.zeros(100)
m_list = np.zeros(100)

for testbee in STEEP:
    vel = calc_vel(testbee)
    T = T_remove_NaN(testbee)

    T_bin = np.arange(min(T), max(T), 0.1)
    bin_vel_LIST = []
    for i in T_bin:
        bin_vel = 0
        n = 0.000001
        for ii in range(len(vel)):
            if np.abs(T[ii] - i) < 0.1:
                bin_vel += vel[ii]
                n += 1
        bin_vel_LIST.append(bin_vel/n)
    for j in range(len(T_master_bin)):
        for jj in range(len(T_bin)):
            if np.abs(T_master_bin[j]-T_bin[jj]) < 0.1:
                v_mean_list[j] += bin_vel_LIST[jj]
                m_list[j] += 1
    # print(v_mean_list)
    # print(m_list)
    #plt.plot(T_bin, bin_vel_LIST,alpha=0.3)
for v in range(len(v_mean_list)):
    v_mean_list[v] /= m_list[v]
plt.plot(T_master_bin, v_mean_list, label="steep")

T_master_bin = np.arange(27, 37, 0.1)
v_mean_list = np.zeros(100)
m_list = np.zeros(100)

for testbee in STEEPEST:
    vel = calc_vel(testbee)
    T = T_remove_NaN(testbee)

    T_bin = np.arange(min(T), max(T), 0.1)
    bin_vel_LIST = []
    for i in T_bin:
        bin_vel = 0
        n = 0.000001
        for ii in range(len(vel)):
            if np.abs(T[ii] - i) < 0.1:
                bin_vel += vel[ii]
                n += 1
        bin_vel_LIST.append(bin_vel/n)
    for j in range(len(T_master_bin)):
        for jj in range(len(T_bin)):
            if np.abs(T_master_bin[j]-T_bin[jj]) < 0.1:
                v_mean_list[j] += bin_vel_LIST[jj]
                m_list[j] += 1
    # print(v_mean_list)
    # print(m_list)
    #plt.plot(T_bin, bin_vel_LIST,alpha=0.3)
for v in range(len(v_mean_list)):
    v_mean_list[v] /= m_list[v]
plt.plot(T_master_bin, v_mean_list, label="steepest")

T_master_bin = np.arange(27, 37, 0.1)
v_mean_list = np.zeros(100)
m_list = np.zeros(100)

for testbee in control_30_30:
    vel = calc_vel(testbee)
    T = T_remove_NaN(testbee)

    T_bin = np.arange(min(T), max(T), 0.1)
    bin_vel_LIST = []
    for i in T_bin:
        bin_vel = 0
        n = 0.000001
        for ii in range(len(vel)):
            if np.abs(T[ii] - i) < 0.1:
                bin_vel += vel[ii]
                n += 1
        bin_vel_LIST.append(bin_vel/n)
    for j in range(len(T_master_bin)):
        for jj in range(len(T_bin)):
            if np.abs(T_master_bin[j]-T_bin[jj]) < 0.1:
                v_mean_list[j] += bin_vel_LIST[jj]
                m_list[j] += 1
    # print(v_mean_list)
    # print(m_list)
    #plt.plot(T_bin, bin_vel_LIST,alpha=0.3)
for v in range(len(v_mean_list)):
    v_mean_list[v] /= m_list[v]
plt.plot(T_master_bin, v_mean_list, label="control 30 30")

T_master_bin = np.arange(27, 37, 0.1)
v_mean_list = np.zeros(100)
m_list = np.zeros(100)
for testbee in control_36_36:
    vel = calc_vel(testbee)
    T = T_remove_NaN(testbee)

    T_bin = np.arange(min(T), max(T), 0.1)
    bin_vel_LIST = []
    for i in T_bin:
        bin_vel = 0
        n = 0.000001
        for ii in range(len(vel)):
            if np.abs(T[ii] - i) < 0.1:
                bin_vel += vel[ii]
                n += 1
        bin_vel_LIST.append(bin_vel/n)
    for j in range(len(T_master_bin)):
        for jj in range(len(T_bin)):
            if np.abs(T_master_bin[j]-T_bin[jj]) < 0.1:
                v_mean_list[j] += bin_vel_LIST[jj]
                m_list[j] += 1
    # print(v_mean_list)
    # print(m_list)
    #plt.plot(T_bin, bin_vel_LIST,alpha=0.3)
for v in range(len(v_mean_list)):
    v_mean_list[v] /= m_list[v]
plt.plot(T_master_bin, v_mean_list, label="control 36 36")
plt.legend()#loc="lower left")
plt.show()


    #k, d = np.polyfit(T, vel, 1)
    #k_list.append(k)
    #d_list.append(d)
    #T_range = np.linspace(min(T), max(T))
    #plt.plot(T_range, T_range * k + d, linestyle='--', color='red', alpha=0.3)

    #plt.xscale('log')
    #plt.yscale('log')
    #plt.xlabel('Temperature[°C]')
    #plt.ylabel('velocity[cm/s]')
    #plt.title(testbee)
    #plt.savefig("FIG_vel_vs_T/" + str(testbee))
    #plt.close()
    # plt.scatter(calc_vel(testbee), calc_delta_T(testbee))
    # plt.show()
