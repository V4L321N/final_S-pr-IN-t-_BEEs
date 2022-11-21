import pandas as pd
import numpy as np
from scipy import stats
import scipy.stats
from scipy.stats import norm
from math import log, floor
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib.patches as patches

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
def calc_vel_shot(item):
    list_vel = []
    list_shot = []
    X_dataset_wo_NAN = X_remove_NaN(item) #test_x
    Y_dataset_wo_NAN = Y_remove_NaN(item) #test_y
    n = len(X_dataset_wo_NAN) - 1
    for t in range(n):
        velocity = np.sqrt((X_dataset_wo_NAN[t + 1] - X_dataset_wo_NAN[t]) ** 2 + (Y_dataset_wo_NAN[t + 1] - Y_dataset_wo_NAN[t]) ** 2)
        if velocity <= 0.5:
            list_shot.append(0)
        else:
            list_shot.append(1)
        list_vel.append(velocity)
    return list_vel, list_shot
#print(calc_vel(testbee))
#plt.show()
"""-end--------calculate duration stopping-----------------"""

# for testbee in head:
#     print()

IB_1 = "BT06A-2"
IB_2 = "BT06A-3"
IB_3 = "BT02B-2"
IB_4 = "BT02B-1"
GF_1 = "BT09A-2"
GF_2 = "BT09B-2"
GF_3 = "BT09B-4"
GF_4 = "BT12B-2"
first_squad = [IB_1, IB_2, IB_3, IB_4, GF_1, GF_2, GF_3, GF_4]
WF_1 = "BT04B-3"
WF_2 = "BT09B-1"
WF_3 = "BT12B-1"
WF_4 = "BT13B-3"
RW_1 = "BT03A-1"
RW_2 = "BT06B-1"
RW_3 = "BT13A-3"
RW_4 = "BT12A-1"
second_squad = [WF_1, WF_2, WF_3, WF_4, RW_1, RW_2, RW_3, RW_4]

fig, ((IB1, IB2, IB3, IB4, GF1, GF2, GF3, GF4)) = plt.subplots(8, 1, figsize=(10,10))
itemize = [IB1, IB2, IB3, IB4, GF1, GF2, GF3, GF4]
type = [r'$IB_1$', r'$IB_2$', r'$IB_3$', r'$IB_4$', r'$GF_1$', r'$GF_2$', r'$GF_3$', r'$GF_4$']
m=0
fig.set_tight_layout(True)
for item in itemize:
    vel, shot = calc_vel_shot(first_squad[m])
    color_RTS = 'tab:red'
    item.set_ylabel('RTS', color=color_RTS, fontdict=dict(weight='bold'))
    item.plot(shot, color=color_RTS)
    ax2 = item.twinx()
    color_vel = 'tab:blue'
    ax2.set_ylabel('v'+r'$ [cm/s]$', color=color_vel, fontdict=dict(weight='bold'))
    ax2.plot(vel, color=color_vel)
    ax2.set_ylim(-0.2,6.2)
    ax2.set_yticks([0,3,6])
    fig.tight_layout()
    item.set_xlim(-1,1201)
    item.set_ylim(-0.04, 1.04)
    item.set_xticks([])
    item.set_yticks([0,1])
    if item == GF4:
        item.set_xticks([0,1200])
        item.set_xlabel('time '+r'$ [s]$')
    item.annotate(type[m], xy=(0.96,0.8),xycoords='axes fraction', fontsize=10)
    m+=1
plt.show()

fig, ((WF1, WF2, WF3, WF4, RW1, RW2, RW3, RW4)) = plt.subplots(8, 1, figsize=(10,10))
itemize = [WF1, WF2, WF3, WF4, RW1, RW2, RW3, RW4]
type = [r'$WF_1$', r'$WF_2$', r'$WF_3$', r'$WF_4$', r'$RW_1$', r'$RW_2$', r'$RW_3$', r'$RW_4$']
n=0
fig.set_tight_layout(True)

for item in itemize:
    vel, shot = calc_vel_shot(second_squad[n])
    color_RTS = 'tab:red'
    item.set_ylabel('RTS', color=color_RTS, fontdict=dict(weight='bold'))
    item.plot(shot, color=color_RTS)
    ax2 = item.twinx()
    color_vel = 'tab:blue'
    ax2.set_ylabel('v'+r'$ [cm/s]$', color=color_vel, fontdict=dict(weight='bold'))
    ax2.plot(vel, color=color_vel)
    ax2.set_ylim(-0.2,6.2)
    ax2.set_yticks([0,3,6])
    fig.tight_layout()
    item.set_xlim(-1,1201)
    item.set_ylim(-0.04, 1.04)
    item.set_xticks([])
    item.set_yticks([0,1])
    if item == RW4:
        item.set_xticks([0,1200])
        item.set_xlabel('time '+r'$ [s]$')
    item.annotate(type[n], xy=(0.96,0.8),xycoords='axes fraction', fontsize=10)
    n+=1
plt.show()


    # plt.plot(calc_vel_shot(testbee)[0])
    # plt.title(testbee)
    # plt.xlabel('time '+r'$ [s]$')
    # plt.ylabel('velocity '+r'$ [cm/s]$')
    # plt.show()

# fig, ax1 = plt.subplots()
#
# color = 'tab:red'
# ax1.set_xlabel('time (s)')
# ax1.set_ylabel('exp', color=color)
# ax1.plot(t, data1, color=color)
# ax1.tick_params(axis='y', labelcolor=color)
#
# ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
#
# color = 'tab:blue'
# ax2.set_ylabel('sin', color=color)  # we already handled the x-label with ax1
# ax2.plot(t, data2, color=color)
# ax2.tick_params(axis='y', labelcolor=color)
#
# fig.tight_layout()  # otherwise the right y-label is slightly clipped
# plt.show()
