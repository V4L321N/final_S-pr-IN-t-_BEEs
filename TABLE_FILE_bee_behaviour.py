import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.patches as patches
from scipy.optimize import curve_fit
from scipy.stats import norm
#import time

ArenaX = pd.DataFrame(data=pd.read_csv("~/Desktop/BAC_bee_mobility/final_S(pr)IN(t)_BEEs/V_3/data_bee_types/Arena.csv"), columns=["ArenaX"])
ArenaY = pd.DataFrame(data=pd.read_csv("~/Desktop/BAC_bee_mobility/final_S(pr)IN(t)_BEEs/V_3/data_bee_types/Arena.csv"), columns=["ArenaY"])
GradientX = pd.DataFrame(data=pd.read_csv("~/Desktop/BAC_bee_mobility/final_S(pr)IN(t)_BEEs/V_3/data_bee_types/Arena.csv"), columns=["GradX"])
GradientY = pd.DataFrame(data=pd.read_csv("~/Desktop/BAC_bee_mobility/final_S(pr)IN(t)_BEEs/V_3/data_bee_types/Arena.csv"), columns=["GradY"])

head = list(pd.DataFrame(data=pd.read_csv("~/Desktop/BAC_bee_mobility/final_S(pr)IN(t)_BEEs/V_3/data_bee_types/LS_spatial_D_x.csv")))

control_30_30 = ["BT17A-1", "BT17A-2", "BT17A-3", "BT17A-4", "BT17B-1", "BT17B-2", "BT17B-3", "BT17B-4"]
control_36_36 = ["BT07A-1", "BT07A-2", "BT07A-3", "BT07A-4", "BT07B-1", "BT07B-2", "BT07B-3", "BT07B-4"]

control_ALL = control_30_30 + control_36_36
for item in control_ALL:
    head.remove(item)

NARROW = ["BT01A-1","BT01A-2","BT01A-3","BT01A-4","BT01C-1","BT01C-2","BT01C-3","BT01C-4","BT02A-1","BT02A-2","BT02A-3","BT02A-4","BT02B-1","BT02B-2","BT02B-3","BT02B-4","BT03A-1","BT03A-2","BT03A-3","BT03A-4","BT03B-1","BT03B-2","BT03B-3","BT03B-4","BT06A-1","BT06A-2","BT06A-3","BT06A-4","BT06B-1","BT06B-2","BT06B-3","BT06B-4","BT18A-1","BT18A-2","BT18A-3","BT18A-4","BT18B-1","BT18B-2","BT18B-3","BT18B-4"]
STEEP = ["BT04A-1","BT04A-2","BT04A-3","BT04A-4","BT04B-1","BT04B-2","BT04B-3","BT04B-4","BT05A-1","BT05A-2","BT05A-3","BT05A-4","BT05B-1","BT05B-2","BT05B-3","BT05B-4","BT08A-1","BT08A-2","BT08A-3","BT08A-4","BT08B-1","BT08B-2","BT08B-3","BT08B-4","BT09A-1","BT09A-2","BT09A-3","BT09A-4","BT09B-1","BT09B-2","BT09B-3","BT09B-4","BT10A-1","BT10A-2","BT10A-3","BT10A-4","BT10B-1","BT10B-2","BT10B-3","BT10B-4"]
STEEPEST = ["BT11A-1","BT11A-2","BT11A-3","BT11A-4","BT11B-1","BT11B-2","BT11B-3","BT11B-4","BT12A-1","BT12A-2","BT12A-3","BT12A-4","BT12B-1","BT12B-2","BT12B-3","BT12B-4","BT13A-1","BT13A-2","BT13A-3","BT13A-4","BT13B-1","BT13B-2","BT13B-3","BT13B-4","BT14A-1","BT14A-2","BT14A-3","BT14A-4","BT14B-1","BT14B-2","BT14B-3","BT14B-4","BT15A-1","BT15A-2","BT15A-3","BT15A-4","BT15B-1","BT15B-2","BT15B-3","BT15B-4"]

"""-begin------function: remove empty entires XY and T-----------------------"""
def X_remove_NaN(item):
    X_dataset_entropy = pd.DataFrame(data=pd.read_csv("~/Desktop/BAC_bee_mobility/final_S(pr)IN(t)_BEEs/V_3/data_bee_types/LS_spatial_D_x.csv"), columns=[item])
    X_dataset_wo_NAN = []
    for m in range(len(X_dataset_entropy)):
        check = X_dataset_entropy.iloc[m].item()
        if str(check) != "nan":
            X_dataset_wo_NAN.append(check)
    return X_dataset_wo_NAN

def Y_remove_NaN(item):
    Y_dataset_entropy = pd.DataFrame(data=pd.read_csv("~/Desktop/BAC_bee_mobility/final_S(pr)IN(t)_BEEs/V_3/data_bee_types/LS_spatial_D_y.csv"), columns=[item])
    Y_dataset_wo_NAN = []
    for m in range(len(Y_dataset_entropy)):
        check = Y_dataset_entropy.iloc[m].item()
        if str(check) != "nan":
            Y_dataset_wo_NAN.append(check)
    return Y_dataset_wo_NAN

def T_remove_NaN(item):
    T_dataset = pd.DataFrame(data=pd.read_csv("~/Desktop/BAC_bee_mobility/final_S(pr)IN(t)_BEEs/V_3/data_bee_types/temperature_runs.csv"), columns=[item])
    T_dataset_wo_NAN = []
    for m in range(1, len(T_dataset)):
        check = T_dataset.iloc[m].item()
        if str(check) != "nan":
            T_dataset_wo_NAN.append(check)
    return T_dataset_wo_NAN
"""-end--------function: remove empty entires XY and T-----------------------"""


"""-begin----------function: different fitting functions---------------------"""
def myExpFunc(x, k1, k2):
    return x ** k1 + k2

# def myExpFunc(x, k1, k2):     # only for comparison with function above
#     return x * np.exp(k1) + k2

def myLinFunc(x, b, m):
    return x * m + b

def myLin2Func(x, m):
    return m * x

def myPower(x,c1,c2,c3):
    return c3 * x ** (c1) + c2

def myPoly(x,q1,q2,q3):
    return x * q1 + x**2 * q2**2 + q3

# def myCurve(x, tau, beta):    # only for comparison with function below
#     return x ** tau * beta

def myCurve(x,tau,beta):
    return tau * np.exp(-beta * x)

def myCurve1(x,tau,beta):
    return  tau / (1 + (beta * x))

def myCurve2(x,tau,beta):
    return  tau * np.exp(-beta * x)

def myCurve3(x,tau,beta,gamma):
    return tau * (beta * x) ** gamma

def powerCurve(f, alpha):
    return 1/(f**alpha)
"""-end------------function: different fitting functions---------------------"""

"""-begin--------function: calculate TSMD------------------------------------"""
def MSD_temporal(item):
    x,y = X_remove_NaN(item),Y_remove_NaN(item)
    MSD = []
    length = len(x)
    for tau in range(1, length):
        MSD_tau = 0
        for t in range(0, (length-tau)):
            MSD_tau += (x[t+tau]-x[t])**2 + (y[t+tau]-y[t])**2
        MSD.append(MSD_tau/(length-tau))
    return MSD
"""-end-----------function: calculate TSMD-----------------------------------"""
#
"""-begin--------list: TSMD ALL BEES-----------------------------------------"""
list_TSMD_b = []
list_TSMD_m = []
for testbee in head:
    listy = MSD_temporal(testbee)
    listx = np.linspace(1, len(listy), len(listy))
    listyCUT = listy[0:20]
    listxCUT = np.linspace(1, len(listyCUT), len(listyCUT))
    popt, pcov = curve_fit(myLinFunc, listxCUT, listyCUT)
    b = str(round(popt[0], 3))
    m = str(round(popt[1], 3))
    list_TSMD_b.append(b)
    list_TSMD_m.append(m)
"""-end----------list: TSMD ALL BEES-----------------------------------------"""
#
"""-begin-----------list: TSMD DIFFERENT FITS ALL BEES/LOGLOG PLOT-----------"""
list_TSMD_v = []
list_TSMD_K = []
list_TSMD_alpha = []
for testbee in head:
    listy = MSD_temporal(testbee)
    listx = np.linspace(0, len(listy), len(listy))

    START_poly = 0#20
    END_poly = 10#len(listy)
    listyPOLY = listy[START_poly:END_poly]
    listxPOLY = np.linspace(START_poly, END_poly, END_poly-START_poly)
    popt_poly, pcov_poly = curve_fit(myPoly, listxPOLY, listyPOLY, maxfev = 2000000, p0=(1, 350, 1))

    START_power = 0#20
    END_power = 10#len(listy)
    listyPOWER = listy[START_power:END_power]
    listxPOWER = np.linspace(START_power, END_power, END_power-START_power)
    popt_power, pcov_power = curve_fit(myPower, listxPOWER, listyPOWER, maxfev = 2000000, p0=(1, 350, 1))

    START_lin = 0#20
    END_lin = 20#len(listy)
    listyLIN = listy[START_lin:END_lin]
    listxLIN = np.linspace(START_lin, END_lin, END_lin-START_lin)
    popt_lin, pcov_lin = curve_fit(myLin2Func, listxLIN, listyLIN, maxfev = 200000, p0=(1))

    list_TSMD_v.append(round(popt_poly[1],2))
    list_TSMD_K.append(round(popt_power[2],2))
    list_TSMD_alpha.append(round(popt_power[0],2))
"""-end-------------list: TSMD DIFFERENT FITS ALL BEES/LOGLOG PLOT-----------"""

"""-begin----------function: calculate sitting and walking velocites------"""
def calc_vel_sw(item):
    list_vel = []
    X_dataset_wo_NAN = X_remove_NaN(item) #test_x
    Y_dataset_wo_NAN = Y_remove_NaN(item) #test_y
    n = len(X_dataset_wo_NAN) - 1
    timeline = np.linspace(0, n, n)
    for t in range(n):
        velocity = np.sqrt((X_dataset_wo_NAN[t + 1] - X_dataset_wo_NAN[t]) ** 2 + (Y_dataset_wo_NAN[t + 1] - Y_dataset_wo_NAN[t]) ** 2)
        list_vel.append(velocity)
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
"""-end----------function: calculate mean sitting and walking velocites------"""

"""-begin--------list: HISTOGRAM OF THE VELOCITIES ALL BEES-----------------"""
list_mean_walk_vel = []
for testbee in head:
    std_walk = np.std(calc_vel_sw(testbee)[0], ddof=1)
    mean_walk = np.mean(calc_vel_sw(testbee)[0])
    list_mean_walk_vel.append(round(mean_walk, 2))
"""-end--------list: HISTOGRAM OF THE VELOCITIES ALL BEES-----------------"""

"""-begin------function: calculate velocity for each time step---------------"""
def calc_vel(item):
    list_vel = []
    X_dataset_wo_NAN = X_remove_NaN(item) #test_x
    Y_dataset_wo_NAN = Y_remove_NaN(item) #test_y
    n = len(X_dataset_wo_NAN) - 1
    for t in range(n):
        velocity = np.sqrt((X_dataset_wo_NAN[t + 1] - X_dataset_wo_NAN[t]) ** 2 + (Y_dataset_wo_NAN[t + 1] - Y_dataset_wo_NAN[t]) ** 2)
        list_vel.append(velocity)
    return list_vel
"""-end--------function: calculate velocity in for each time step------------"""

"""-begin--------function: calculate fourier transform of the velocity-------"""
def velocity_FFT(item):
    return np.fft.fft(calc_vel(item), norm="ortho") #either norm="ortho" here or divided by C_2 in PSD()
"""-end----------function: calculate fourier transform of the velocity-------"""

"""-begin--------function: calculate power spectral density of the FT--------"""
def PSD_vel(item):
    vel_FFT = velocity_FFT(item)
    loop_L = round(len(vel_FFT)/2)
    PSD_list = []
    for i in range(loop_L):
        C_2 = 1
        absolute = C_2 * (vel_FFT[i] * np.conj(vel_FFT[i]))
        PSD_list.append(absolute.real)
    return PSD_list
"""-end----------function: calculate power spectral density of the angle FT--"""

"""-begin--------function: analytical solution for the PSD-------------------"""
def S(w, D, coupling_a):
    return 2 * D / (coupling_a ** 2 + w ** 2)
"""-end----------function: analytical solution for the PSD-------------------"""

"""-begin--------list: PSD OF THE VELOCITY ALL BEES-------------------------"""
list_D_v = []
list_a_v = []
for testbee in head:
    psd_test = PSD_vel(testbee)
    length = len(psd_test)
    start = 0.001
    end = 2 * np.pi / 2
    omega = np.linspace(start, end, length)
    popt, pcov = curve_fit(S, omega, psd_test)
    list_D_v.append(round(popt[0], 2))
    list_a_v.append(round(popt[1], 2))
"""-end----------list: PSD OF THE VELOCITY ALL BEES-------------------------"""

"""-begin------function: calculate turning angle in for each time step-------"""
def calc_theta(item):
    list_theta = []
    X_dataset_wo_NAN = X_remove_NaN(item) #test_x
    Y_dataset_wo_NAN = Y_remove_NaN(item) #test_y
    n = len(X_dataset_wo_NAN) - 1
    timeline = np.linspace(0, n, n)
    for t in range(n):
        nom = Y_dataset_wo_NAN[t + 1] - Y_dataset_wo_NAN[t]
        den = X_dataset_wo_NAN[t + 1] - X_dataset_wo_NAN[t]
        theta = (np.arctan2(nom, den))
        list_theta.append(theta)
    return list_theta
"""-end--------function: calculate turning angle in for each time step-------"""

"""-begin------function: position of the gradient depending on bee position--"""
# Coordinates of the optimum of the Gradient "G" lie at (0, 30)
def calc_dir_G(item):
    G_x = 0
    G_y = 30
    list_G = []
    X_dataset_wo_NAN = X_remove_NaN(item)
    Y_dataset_wo_NAN = Y_remove_NaN(item)
    n = len(X_dataset_wo_NAN) - 1
    timeline = np.linspace(0, n, n)
    for t in range(n):
        nom = (Y_dataset_wo_NAN[t + 1] + Y_dataset_wo_NAN[t]) - G_y
        den = (X_dataset_wo_NAN[t + 1] + X_dataset_wo_NAN[t]) - G_x
        theta_G = (np.arctan2(nom, den))
        list_G.append(theta_G)
    return list_G
"""-end--------function: position of the gradient depending on bee position--"""

"""-begin------function: turning angle in respect of gradient position-------"""
def theta_in_G(item):
    list_theta_in_G = []
    theta = calc_theta(item)
    dir_G = calc_dir_G(item)
    n = len(dir_G)
    for t in range(n):
        list_theta_in_G.append(theta[t] - dir_G[t])
    return list_theta_in_G
"""-end--------function: turning angle in respect of gradient position-------"""

"""-begin------function: check turning angle to avoid jumps from -pi and +pi-"""
def recalc_w_2pi(item):
    theta_G = theta_in_G(item)
    n = len(theta_G)
    for t in range(n):
        while theta_G[t] - theta_G[t-1] > np.pi:
            theta_G[t] -= 2 * np.pi
        while theta_G[t] - theta_G[t-1] < -np.pi:
            theta_G[t] += 2 * np.pi
    return theta_G
"""-end--------function: check turning angle to avoid jumps from -pi and +pi-"""

"""-begin--------function: calculate fourier transform of the turning angle--"""
def theta_FFT(item):
    return np.fft.fft(recalc_w_2pi(item), norm="ortho")
"""-end----------function: calculate fourier transform of the turning angle--"""

"""-begin--------function: calculate power spectral density of the FT--------"""
def PSD_theta(item):
    th_FFT = theta_FFT(item)
    loop_L = round(len(th_FFT)/2)
    PSD_list = []
    for i in range(loop_L):
        C_1 = 1#/len(theta_FFT)
        absolute = C_1 * (th_FFT[i] * np.conj(th_FFT[i]))
        PSD_list.append(absolute.real)
    return PSD_list
"""-end----------function: calculate power spectral density of the FT--------"""

"""-begin--------list: PSD OF ANGLE THETA ALL BEES--------------------------"""
list_D_r = []
list_a_r = []
for testbee in head:
    psd_test = PSD_theta(testbee)
    length = len(psd_test)
    start = 0.001
    end = 2 * np.pi / 2
    omega = np.linspace(start, end, length)
    popt, pcov = curve_fit(S, omega, psd_test, maxfev = 200000, p0=(10,0.001))
    list_D_r.append(round(popt[0], 2))
    list_a_r.append(round(popt[1], 2))
"""-end----------list: PSD OF ANGLE THETA ALL BEES--------------------------"""

"""-begin------function: map velocity as shot noise--------------------------"""
def calc_vel_shot(item):
    list_vel = []
    list_shot = []
    X_dataset_wo_NAN = X_remove_NaN(item)
    Y_dataset_wo_NAN = Y_remove_NaN(item)
    n = len(X_dataset_wo_NAN) - 1
    for t in range(n):
        velocity = np.sqrt((X_dataset_wo_NAN[t + 1] - X_dataset_wo_NAN[t]) ** 2 + (Y_dataset_wo_NAN[t + 1] - Y_dataset_wo_NAN[t]) ** 2)
        if velocity <= 0.5:
            list_shot.append(0)
        else:
            list_shot.append(1)
        list_vel.append(velocity)
    return list_vel, list_shot
"""-end--------function: map velocity as shot noise--------------------------"""

"""-begin------function: calculate walking duration--------------------------"""
def calc_dur_walk(item):
    list_vel = []
    X_dataset_wo_NAN = X_remove_NaN(item)
    Y_dataset_wo_NAN = Y_remove_NaN(item)
    n = len(X_dataset_wo_NAN) - 1
    for t in range(n):
        velocity = np.sqrt((X_dataset_wo_NAN[t + 1] - X_dataset_wo_NAN[t]) ** 2 + (Y_dataset_wo_NAN[t + 1] - Y_dataset_wo_NAN[t]) ** 2)
        list_vel.append(velocity)
    T_dataset_wo_NAN = T_remove_NaN(item)
    T_minmax_data = pd.DataFrame(data=pd.read_csv("~/Desktop/BAC_bee_mobility/final_S(pr)IN(t)_BEEs/V_3/data_bee_types/temperature_variables.csv"), columns=[item])
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
"""-end--------function: calculate walking duration--------------------------"""

"""-begin------function: calculate stopping duration-------------------------"""
def calc_dur_stop(item):
    list_vel = []
    X_dataset_wo_NAN = X_remove_NaN(item)
    Y_dataset_wo_NAN = Y_remove_NaN(item)
    n = len(X_dataset_wo_NAN) - 1
    for t in range(n):
        velocity = np.sqrt((X_dataset_wo_NAN[t + 1] - X_dataset_wo_NAN[t]) ** 2 + (Y_dataset_wo_NAN[t + 1] - Y_dataset_wo_NAN[t]) ** 2)
        list_vel.append(velocity)
    T_dataset_wo_NAN = T_remove_NaN(item)
    T_minmax_data = pd.DataFrame(data=pd.read_csv("~/Desktop/BAC_bee_mobility/final_S(pr)IN(t)_BEEs/V_3/data_bee_types/temperature_variables.csv"), columns=[item])
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
"""-end--------function: calculate stopping duration-------------------------"""

"""-begin------function: calculate stopping duration v2----------------------"""
def calc_dur_stop_2(item):
    list_vel = []
    X_dataset_wo_NAN = X_remove_NaN(item)
    Y_dataset_wo_NAN = Y_remove_NaN(item)
    n = len(X_dataset_wo_NAN) - 1
    for t in range(n):
        velocity = np.sqrt((X_dataset_wo_NAN[t + 1] - X_dataset_wo_NAN[t]) ** 2 + (Y_dataset_wo_NAN[t + 1] - Y_dataset_wo_NAN[t]) ** 2)
        list_vel.append(velocity)
    T_dataset_wo_NAN = T_remove_NaN(item)
    T_minmax_data = pd.DataFrame(data=pd.read_csv("~/Desktop/BAC_bee_mobility/final_S(pr)IN(t)_BEEs/V_3/data_bee_types/temperature_variables.csv"), columns=[item])
    T_max = T_minmax_data.iloc[1].item()
    T_min = T_minmax_data.iloc[0].item()
    stop_vel = 0.4
    duration_stop = 0
    stop_temperatures_2 = []
    for v in range(len(list_vel[:-1])):
        if list_vel[v] <= stop_vel:
            duration_stop += 1
            stop_temperatures_2.append(T_dataset_wo_NAN[v])
            if list_vel[v+1] > stop_vel:
                all_stop_durations_2.append(duration_stop)
                mean_delta_stop_temp_2.append(np.abs(np.mean(stop_temperatures_2) - T_max))
                #mean_delta_stop_temp.append(np.abs((np.mean(stop_temperatures) - T_max))/np.abs(T_max - T_min))
                duration_walk = 0
                walk_temperatures = []
            elif v == len(list_vel[:-2]):
                all_stop_durations_2.append(duration_stop)
                mean_delta_stop_temp_2.append(np.abs(np.mean(stop_temperatures_2) - T_max))
                #mean_delta_stop_temp.append(np.abs((np.mean(stop_temperatures) - T_max))/np.abs(T_max - T_min))
    return all_stop_durations_2, mean_delta_stop_temp_2
"""-end--------function: calculate stopping duration v2----------------------"""

"""-begin--------list: STOPPING OVER TEMPERATURE ALL BEES-------------------"""
list_tau = []
list_beta = []
for testbee in head:
    xspace = np.linspace(0,10,99)
    vel = calc_vel(testbee)
    vel_0 = np.mean(vel[0])
    stop_durations, stop_temp = calc_dur_stop(testbee)
    if testbee not in ["BT05A-2","BT06A-3"]:
        # popt1, pcov1 = curve_fit(myCurve1, stop_temp, stop_durations, maxfev = 200000, p0=(0,0))
        popt2, pcov2 = curve_fit(myCurve2, stop_temp, stop_durations, maxfev = 200000, p0=(0,0))
        # popt3, pcov3 = curve_fit(myCurve3, stop_temp, stop_durations, maxfev = 200000, p0=(0,0,0))
        list_tau.append(round(popt2[0], 2))
        list_beta.append(round(popt2[1], 2))
    else:
        list_tau.append(0)
        list_beta.append(0)
"""-end----------list: STOPPING OVER TEMPERATURE ALL BEES-------------------"""

"""-begin------function: calculate probability at a temperature--------------"""
def probability_T(item):
    T_dataset_wo_NAN = T_remove_NaN(item)
    T_minmax_data = pd.DataFrame(data=pd.read_csv("data_bee_types/temperature_variables.csv"), columns=[item])
    T_max = T_minmax_data.iloc[1].item()
    prob_dataset = []
    for T in range(len(T_dataset_wo_NAN)):
        prob_dataset.append(np.abs(T_dataset_wo_NAN[T] - T_max))
    return prob_dataset
"""-end--------function: calculate probability at a temperature--------------"""

"""-begin------function: calculate velocity at stopping events---------------"""
def calc_vel_at_noonly_stopping(item):
    list_vel = []
    X_dataset_wo_NAN = X_remove_NaN(item)
    Y_dataset_wo_NAN = Y_remove_NaN(item)
    n = len(X_dataset_wo_NAN) - 1
    timeline = np.linspace(0, n, n)
    for t in range(n):
        velocity = np.sqrt((X_dataset_wo_NAN[t + 1] - X_dataset_wo_NAN[t]) ** 2 + (Y_dataset_wo_NAN[t + 1] - Y_dataset_wo_NAN[t]) ** 2)
        list_vel.append(velocity)
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
"""-end--------function: calculate velocity at stopping events---------------"""

"""-begin--------figure: PSD OF RTS 16 BEES----------------------------------"""
list_alpha_1 = []
list_alpha_2 = []
start = 0
end = 6
dT = 0.1
xspace = np.linspace(0, 10, 1000)
for testbee in head:
    if testbee not in ["BT05A-2","BT06A-3","BT14A-2"]:
        vel_PSD = PSD_vel(testbee)
        length = len(vel_PSD)
        start_new = 0.001
        end_new = 2 * np.pi / 2
        omega_new = np.linspace(start_new, end_new, length)
        mean_velocity = np.mean(calc_vel_at_noonly_stopping(testbee)[0])
        all_stop_durations, mean_delta_stop_temp = calc_dur_stop(testbee)
        all_walk_durations, mean_delta_walk_temp = calc_dur_walk(testbee)
        print(testbee)
        mean_t1 = np.mean(all_walk_durations)

        #popt_stop1, pcov_stop1 = curve_fit(myCurve1, mean_delta_stop_temp, all_stop_durations, maxfev = 200000, p0=(0,0))#, bounds=([0,0], [3.0, 1.0]))
        popt_stop2, pcov_stop2 = curve_fit(myCurve2, mean_delta_stop_temp, all_stop_durations, maxfev = 200000, p0=(0,0))#, bounds=([0,0], [3.0, 1.0]))
        #popt_stop3, pcov_stop3 = curve_fit(myCurve3, mean_delta_stop_temp, all_stop_durations, maxfev = 200000, p0=(0,0,0))#, bounds=([0,0,0], [3.0, 1.0, 1.0]))
        prob_dataset = probability_T(testbee)
        bins = np.arange(start, end, dT)
        rangeT = np.linspace(start, end, 59)
        data_entries, bins = np.histogram(prob_dataset, bins=bins, range=(start,end), density=True)
        def S(w, v_0, t_1, Prob, dTemp):
            sum = 0
            for T in np.arange(0,5.9,0.1):
                sum += (Prob[int(round(T/0.1, 0))] * dTemp) / ((myCurve2(T, *popt_stop2) + 1) * ((1 / myCurve2(T, *popt_stop2) + 1 / t_1) ** 2 + w ** 2))
            return 2 * v_0 ** 2 * sum
        omega = np.linspace(0.0001, (np.pi/2))
        x_low = 0.1
        x_high = 1.0
        y_low = S(x_low, mean_velocity, mean_t1, data_entries, dT)
        y_high = S(x_high, mean_velocity, mean_t1, data_entries, dT)
        alpha = np.abs(np.log10(y_high/y_low))
        list_alpha_1.append(round(alpha,2))

        #high_omega = np.linspace(0.15,1.5)
        high_omega = np.linspace(0.015,1.5)
        popt_pow, pcov_pow = curve_fit(powerCurve, high_omega, S(high_omega, mean_velocity, mean_t1, data_entries, dT), maxfev=200000, p0=(0))
        list_alpha_2.append(round(popt_pow[0],2))

        print(testbee)
    else:
        list_alpha_1.append(0)
        list_alpha_2.append(0)
"""-end----------figure: PSD OF RTS 16 BEES----------------------------------"""

"""-begin------function: calculate gradient----------------------------------"""
def calc_delta_T(item):
    calc_dataset = T_remove_NaN(item)
    delta_T = []
    for i in range(len(calc_dataset) - 1):
        delta_T.append(calc_dataset[i + 1] - calc_dataset[i])
    return delta_T
"""-end--------function: calculate gradient----------------------------------"""


def type(item):
    type_item = pd.DataFrame(data=pd.read_csv("~/Desktop/BAC_bee_mobility/final_S(pr)IN(t)_BEEs/V_3/data_bee_types/bee_types_manual.csv"), columns=[item])
    return type_item

list_type = []
list_name = []
list_gradient = []
for testbee in head:
    TY = type(testbee).iloc[0].item()
    if TY == "Immobile Bee":
        list_type.append("IB")
    elif TY == "Goal Finder":
        list_type.append("GF")
    elif TY == "Wall Follower":
        list_type.append("WF")
    elif TY == "Random Walker":
        list_type.append("RW")
    list_name.append(testbee)
    if testbee in NARROW:
        list_gradient.append(1)
    elif testbee in STEEP:
        list_gradient.append(2)
    if testbee in STEEPEST:
        list_gradient.append(3)

# fig, ax = plt.subplots(figsize=(7,10))
# # hide axes
# fig.patch.set_visible(False)
# ax.axis('off')
# ax.axis('tight')
#df = pd.DataFrame(list(zip(list_TSMD_b, list_TSMD_m, list_mean_walk_vel)), columns=[r'$b_{TSMD}$', r'$m_{TSMD}$', r'$v_{mean}$'])
#df = pd.DataFrame(list(zip(list_name[0:30], list_type[0:30], list_TSMD_b[0:30], list_TSMD_m[0:30], list_mean_walk_vel[0:30])), columns=["ID", "type", r'$b_{TSMD}$', r'$m_{TSMD}$', r'$v_{mean}$'])
# ax.table(cellText=df.values, colLabels=df.columns, loc='center', fontsize=10)
# fig.tight_layout()
# plt.show()

df = pd.DataFrame(list(zip(list_name, list_type, list_TSMD_b, list_TSMD_m, list_TSMD_v, list_TSMD_K, list_TSMD_alpha, list_mean_walk_vel, list_D_v, list_a_v, list_D_r, list_a_r, list_tau, list_beta, list_alpha_1, list_alpha_2)), columns=["ID", "type", r'$b_{TSMD}$', r'$m_{TSMD}$', r'$v_{TSMD}$', r'$K_{TSMD}$', r'$\alpha_{TSMD}$', r'$v_{mean}$', r'$D_{v}$', r'$a_{v}$', r'$D_{\theta}$', r'$a_{\theta}$', r'$\tau$', r'$\beta$', r'$\alpha_{1}$', r'$\alpha_{2}$'])
df.to_csv("MASTER_TABLE.csv")

#print(list_TSMD_b)

#mean_TSMD_b = np.mean(list_TSMD_b)
#mean_TSMD_m = np.mean(list_TSMD_m)
mean_TSMD_v = np.mean(list_TSMD_v)
mean_TSMD_K = np.mean(list_TSMD_K)
mean_TSMD_alpha = np.mean(list_TSMD_alpha)
mean_mean_walk_vel = np.mean(list_mean_walk_vel)
mean_D_v = np.mean(list_D_v)
mean_a_v = np.mean(list_a_v)
mean_D_r = np.mean(list_D_r)
mean_a_r = np.mean(list_a_r)
mean_tau = np.mean(list_tau)
mean_beta = np.mean(list_beta)
mean_alpha_1 = np.mean(list_alpha_1)
mean_alpha_2 = np.mean(list_alpha_2)

print(mean_TSMD_b, mean_TSMD_m, mean_TSMD_v, mean_TSMD_K, mean_TSMD_alpha, mean_TSMD_alpha, mean_mean_walk_vel, mean_D_v, mean_a_v, mean_D_r, mean_a_r, mean_tau, mean_beta, mean_alpha_1, mean_alpha_2)

std_TSMD_b = np.std(list_TSMD_b)
std_TSMD_m = np.std(list_TSMD_m)
std_TSMD_v = np.std(list_TSMD_v)
std_TSMD_K = np.std(list_TSMD_K)
std_TSMD_alpha = np.std(list_TSMD_alpha)
std_mean_walk_vel = np.std(list_mean_walk_vel)
std_D_v = np.std(list_D_v)
std_a_v = np.std(list_a_v)
std_D_r = np.std(list_D_r)
std_a_r = np.std(list_a_r)
std_tau = np.std(list_tau)
std_beta = np.std(list_beta)
std_alpha_1 = np.std(list_alpha_1)
std_alpha_2 = np.std(list_alpha_2)

print(std_TSMD_b, std_TSMD_m, std_TSMD_v, std_TSMD_K, std_TSMD_alpha, std_TSMD_alpha, std_mean_walk_vel, std_D_v, std_a_v, std_D_r, std_a_r, std_tau, std_beta, std_alpha_1, std_alpha_2)
