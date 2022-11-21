import pandas as pd
import numpy as np
from scipy import stats
import scipy.stats
from scipy.stats import norm
from math import log, floor
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

head = list(pd.DataFrame(data=pd.read_csv("data_bee_types/LS_spatial_D_x.csv")))

IB_1 = "BT06A-2"
IB_2 = "BT06A-3"
IB_3 = "BT02B-2"
IB_4 = "BT02B-1"
GF_1 = "BT09A-2"
GF_2 = "BT09B-2"
GF_3 = "BT09B-4"
GF_4 = "BT12B-2"
WF_1 = "BT04B-3"
WF_2 = "BT09B-1"
WF_3 = "BT12B-1"
WF_4 = "BT13B-3"
RW_1 = "BT03A-1"
RW_2 = "BT06B-1"
RW_3 = "BT13A-3"
RW_4 = "BT12A-1"
Well_Behaved = [IB_1, IB_2, IB_3, IB_4, GF_1, GF_2, GF_3, GF_4, WF_1, WF_2, WF_3, WF_4, RW_1, RW_2, RW_3, RW_4]

#test_x = [60, 50, 40, 30, 20, 10, 0, 10, 20, 30, 40, 50, 60, 50, 40, 30, 20, 10, 0, 10, 20, 30, 40, 50, 60, 50, 40, 30, 20, 10, 0, 10, 20, 30, 40, 50, 60]
#test_y = [30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30]
testbee = "BT01A-1"

""" CONSTANTS """
#--------------------------------------------------------------------------------

# D_r = 400
# a = 0.4

# D_r = 1000
# a = 1

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
    T_min = T_minmax_data.iloc[0].item()
    T_max = T_minmax_data.iloc[1].item()

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
                #mean_delta_stop_temp.append(np.abs(np.mean(stop_temperatures) - T_ideal))
                mean_delta_stop_temp.append(np.abs(np.mean(stop_temperatures) - T_max))#/np.abs(T_max-T_min))
                duration_walk = 0
                walk_temperatures = []
            elif v == len(list_vel[:-2]):
                all_stop_durations.append(duration_stop)
                mean_delta_stop_temp.append(np.abs(np.mean(stop_temperatures) - T_max))#/np.abs(T_max-T_min))
    return all_stop_durations, mean_delta_stop_temp
#print(calc_vel(testbee))
#plt.show()
"""-end--------calculate duration stopping-----------------"""

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

"""-begin------calculate velocity for each time step-----------------"""
# def calc_vel(item):
#     list_vel = []
#     X_dataset_wo_NAN = X_remove_NaN(item) #test_x
#     Y_dataset_wo_NAN = Y_remove_NaN(item) #test_y
#     n = len(X_dataset_wo_NAN) - 1
#     timeline = np.linspace(0, n, n)
#     for t in range(n):
#         velocity = np.sqrt((X_dataset_wo_NAN[t + 1] - X_dataset_wo_NAN[t]) ** 2 + (Y_dataset_wo_NAN[t + 1] - Y_dataset_wo_NAN[t]) ** 2)
#         list_vel.append(velocity)
#     return list_vel
#plt.plot(calc_vel(testbee))
#plt.show()
"""-end--------calculate velocity in for each time step-----------------"""

"""-begin------calculate velocity (except zero) for each time step-----------------"""
def calc_vel(item):
    list_vel = []
    X_dataset_wo_NAN = X_remove_NaN(item) #test_x
    Y_dataset_wo_NAN = Y_remove_NaN(item) #test_y
    l = len(X_dataset_wo_NAN) - 1
    timeline = np.linspace(0, l, l)
    for t in range(l):
        velocity = np.sqrt((X_dataset_wo_NAN[t + 1] - X_dataset_wo_NAN[t]) ** 2 + (Y_dataset_wo_NAN[t + 1] - Y_dataset_wo_NAN[t]) ** 2)
        list_vel.append(velocity)
    #return list_vel

    unique, counts = np.unique(list_vel, return_counts=True)
    minimum = min(counts)

    treshold_list = []

    for u in range(len(unique)):
        if counts[u] <= minimum and unique[u] <= 0.5:
            treshold_list.append(unique[u])

    treshhold = max(treshold_list)
    list_vel_NO_stopping = []
    list_vel_ONLY_stopping = []
    for tt in range(len(list_vel)):
        if list_vel[tt] >= treshhold:
            list_vel_NO_stopping.append(list_vel[tt])
        else:
            list_vel_ONLY_stopping.append(list_vel[tt])

    return list_vel_NO_stopping, list_vel_ONLY_stopping
#plt.plot(calc_vel(testbee))
#plt.show()
"""-end--------calculate velocity  (except zero) in for each time step-----------------"""


"""-begin--------calculate fourier transform of the velocity-----------------"""
def velocity_FFT(item):
    return np.fft.fft(calc_vel(item), norm="ortho") #either norm="ortho" here or divided by C_2 in PSD()
"""-end--------calculate fourier transform of the velocity-------------------"""


"""-begin--------calculate power spectral density of the FT------------------"""
def PSD(item):
    vel_FFT = velocity_FFT(item)
    loop_L = round(len(vel_FFT)/2)
    PSD_list = []
    for i in range(loop_L):
        C_2 = 1
        absolute = C_2 * (vel_FFT[i] * np.conj(vel_FFT[i]))
        PSD_list.append(absolute.real)
    return PSD_list

def RTS_PSD(item):
    return 0

def myCurve1(x,tau,beta):
    return tau * np.exp(-beta * x) #tau / (1 + (beta * x)) #

def myCurve2(x,tau,beta,gamma):
    return tau * (beta * x) ** gamma #tau / (1 + (beta * x)) #

def myCurve3(x,tau,beta):
    return tau / (1 + beta * x)

# def PSD(item):
#     PSD_list = []
#     for i in range(round(len(theta_FFT)/2)):
#         PSD_list.append(theta_FFT[i].real ** 2 + theta_FFT[i].imag ** 2)
#     return PSD_list
"""-end----------calculate power spectral density of the angle FT------------"""

fig, ((IB1, IB2, IB3, IB4), (GF1, GF2, GF3, GF4), (WF1, WF2, WF3, WF4), (RW1, RW2, RW3, RW4)) = plt.subplots(4, 4, figsize=(10,10))
itemize = [IB1, IB2, IB3, IB4, GF1, GF2, GF3, GF4, WF1, WF2, WF3, WF4, RW1, RW2, RW3, RW4]
type = [r'$IB_1$', r'$IB_2$', r'$IB_3$', r'$IB_4$', r'$GF_1$', r'$GF_2$', r'$GF_3$', r'$GF_4$', r'$WF_1$', r'$WF_2$', r'$WF_3$', r'$WF_4$', r'$RW_1$', r'$RW_2$', r'$RW_3$', r'$RW_4$']
n=0
fig.set_tight_layout(True)
for item in itemize:
    item.set_ylim(-20,1300)
    item.set_xlim(-0.2,10.2)
    xspace = np.linspace(0,10,99)
    #xspace = np.arange(0,10,1)
    vel = calc_vel(Well_Behaved[n])
    vel_0 = np.mean(vel[0])
    stop_durations, stop_temp = calc_dur_stop(Well_Behaved[n])
    print(len(stop_durations), len(stop_temp))
    item.scatter(stop_temp, stop_durations, color='black', alpha=0.3)
    item.annotate(type[n], xy=(0.8,0.9),xycoords='axes fraction', fontsize=12)
    item.set_xticks([])
    item.set_yticks([])
    if item != IB2:
        popt1, pcov1 = curve_fit(myCurve1, stop_temp, stop_durations, maxfev = 200000, p0=(0,0))
        item.plot(myCurve1(np.linspace(0,10,11), *popt1), color="red", linestyle='dotted', label='t*e^(-bx)')
        #popt2, pcov2 = curve_fit(myCurve2, stop_temp, stop_durations, maxfev = 200000, p0=(0,0,0))
        #item.plot(myCurve2(np.linspace(0,100,100), *popt2), color="red", linestyle='dashed', label='t*(bx)^g')
        # popt3, pcov3 = curve_fit(myCurve3, stop_temp, stop_durations, maxfev = 200000, p0=(0,0))
        # item.plot(myCurve3(xspace, *popt3), color="red", linestyle='dashdot', label='t/(1+bx)')
    if item == IB2:
        item.annotate('invalid', xy=(0.35,0.45),xycoords='axes fraction', color='red', fontsize=12)
    if item in [RW1, RW2, RW3, RW4]:
        item.set_xticks([0,5,10])
        item.set_xlabel(r'$\Delta T (\degree C)$')
    if item in [IB1, GF1, WF1, RW1]:
        item.set_yticks([0,600,1200])
        item.set_ylabel(r'$t_{s}(s)$')
    n += 1
#plt.legend()
plt.show()
