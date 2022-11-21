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
    T_max = T_minmax_data.iloc[1].item()
    #prob_dataset = []
    for T in range(len(T_dataset_wo_NAN)):
        prob_dataset.append(np.abs(T_dataset_wo_NAN[T] - T_max))
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
    T_max = T_minmax_data.iloc[1].item()
    T_min = T_minmax_data.iloc[0].item()

    stop_vel = 0.4
    duration_stop = 0
    #all_stop_durations = []
    stop_temperatures = []
    #mean_delta_stop_temp = []
    for v in range(len(list_vel[:-1])):
        if list_vel[v] <= stop_vel:
            duration_stop += 1
            stop_temperatures.append(T_dataset_wo_NAN[v])
            if list_vel[v+1] > stop_vel:
                all_stop_durations.append(duration_stop)
                mean_delta_stop_temp.append(np.abs(np.mean(stop_temperatures) - T_max))
                #mean_delta_stop_temp.append(np.abs((np.mean(stop_temperatures) - T_max))/np.abs(T_max - T_min))
                duration_walk = 0
                walk_temperatures = []
            elif v == len(list_vel[:-2]):
                all_stop_durations.append(duration_stop)
                mean_delta_stop_temp.append(np.abs(np.mean(stop_temperatures) - T_max))
                #mean_delta_stop_temp.append(np.abs((np.mean(stop_temperatures) - T_max))/np.abs(T_max - T_min))
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
    #all_walk_durations = []
    walk_temperatures = []
    #mean_delta_walk_temp = []
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

# """-begin------calculate velocity for each time step-----------------"""
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
# #plt.plot(calc_vel(testbee))
# #plt.show()
# """-end--------calculate velocity in for each time step-----------------"""

"""-begin------calculate velocity (except zero) for each time step-----------------"""
def calc_vel(item):
    list_vel = []
    X_dataset_wo_NAN = X_remove_NaN(item) #test_x
    Y_dataset_wo_NAN = Y_remove_NaN(item) #test_y
    n = len(X_dataset_wo_NAN) - 1
    timeline = np.linspace(0, n, n)
    for t in range(n):
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



def myPoly(x, a, b, c):
    return a * x**2 + b * x +c

def fit_function(x, A, beta, B, mu, sigma):
    return (np.sqrt(A**2) * np.exp(-x/beta) + np.sqrt(B**2) * np.exp(-1.0 * (x - mu)**2 / (2 * sigma**2)))

def myCurve1(x,tau,beta):
    return  tau / (1 + (beta * x))#tau * np.exp(-beta * x) #

def myCurve2(x,tau,beta):
    return  tau * np.exp(-beta * x) #

def myCurve3(x,tau,beta,gamma):
    return tau * (beta * x) ** gamma #tau / (1 + (beta * x)) #

# def PSD(item):
#     PSD_list = []
#     for i in range(round(len(theta_FFT)/2)):
#         PSD_list.append(theta_FFT[i].real ** 2 + theta_FFT[i].imag ** 2)
#     return PSD_list
"""-end----------calculate power spectral density of the angle FT------------"""

#fig, ((IB1, IB2, IB3, IB4), (GF1, GF2, GF3, GF4), (WF1, WF2, WF3, WF4), (RW1, RW2, RW3, RW4)) = plt.subplots(4, 4, figsize=(10,10))
fig, ((IB1,GF1),(WF1,RW1)) = plt.subplots(2, 2, figsize=(7,7))
#itemizeIB = [IB1, IB2, IB3, IB4]#, GF1, GF2, GF3, GF4, WF1, WF2, WF3, WF4, RW1, RW2, RW3, RW4]
#type = [r'$IB_1$', r'$IB_2$', r'$IB_3$', r'$IB_4$', r'$GF_1$', r'$GF_2$', r'$GF_3$', r'$GF_4$', r'$WF_1$', r'$WF_2$', r'$WF_3$', r'$WF_4$', r'$RW_1$', r'$RW_2$', r'$RW_3$', r'$RW_4$']
type = [r'$IB_{1-4}$', r'$GF_{1-4}$', r'$WF_{1-4}$', r'$RW_{1-4}$']
IB_group_STOPS_at_dT = []
IB_group_TEMPS_at_stop = []
n_IB=0
fig.set_tight_layout(True)
all_stop_durations = []
mean_delta_stop_temp = []
all_walk_durations = []
mean_delta_walk_temp = []
prob_dataset = []

for item in range(0,4,1):#itemizeIB:
    calc_dur_stop(Well_Behaved[n_IB])
    probability_T(Well_Behaved[n_IB])
    calc_dur_walk(Well_Behaved[n_IB])
    n_IB += 1

mean_t1 = np.mean(all_walk_durations)
start = 0
end = 6
dT = 0.1
v_IB = 0.607
bins = np.arange(start, end, dT)
rangeT = np.linspace(start, end, 59)
#nr_bins = 61
#bins = np.linspace(start, end, nr_bins)
data_entries_IB, bins_IB = np.histogram(prob_dataset, bins=bins, range=(start,end), density=True)
#IB1.hist(prob_dataset, bins, density=True)
#IB1.plot(rangeT, data_entries_IB)
IB1.annotate(r'$IB_{1-4}$', xy=(0.73,0.9),xycoords='axes fraction', fontsize=12)

#popt_stop1, pcov_stop1 = curve_fit(myCurve1, mean_delta_stop_temp, all_stop_durations, maxfev = 200000, p0=(0,0))#, bounds=([0,0], [3.0, 1.0]))
popt_stop2, pcov_stop2 = curve_fit(myCurve2, mean_delta_stop_temp, all_stop_durations, maxfev = 200000, p0=(0,0))#, bounds=([0,0], [3.0, 1.0]))
#popt_stop3, pcov_stop3 = curve_fit(myCurve3, mean_delta_stop_temp, all_stop_durations, maxfev = 200000, p0=(0,0,0))#, bounds=([0,0,0], [3.0, 1.0, 1.0]))
#xspace = np.linspace(0, 10, 1000)
#IB1.set_ylim(-10,1300)
#IB1.set_xlim(-0.01,8.01)
#IB1.plot(xspace, myCurve1(xspace, *popt_stop1), color='red', linewidth=1.1, linestyle='dashed', label=r'$ \tau / (1 + (\beta * \Delta T)) $')
#IB1.plot(xspace, myCurve2(xspace, *popt_stop2), color='red', linewidth=1.1, linestyle='dashdot', label=r'$ \tau * \exp(-\beta * \Delta T) $')
#IB1.plot(xspace, myCurve3(xspace, *popt_stop3), color='firebrick', linewidth=1.1, label=r'$ \tau * (\beta * \Delta T) ** \gamma $')
#IB1.scatter(mean_delta_stop_temp, all_stop_durations, color="black", alpha=0.3)
#IB1.set_ylabel(r'$t_{s}(s)$')

def RTS_PSD(w):#, v_0, P, t_0, t_1, range, dT):
    sum = 0
    for T in np.arange(0,5.9,0.1):
        sum += (data_entries_IB[int(round(T/0.1, 0))] * dT) / ((myCurve2(T, *popt_stop2) + 1) * ((1 / myCurve2(T, *popt_stop2) + 1 / mean_t1) ** 2 + w ** 2))
    return v_IB ** 2 * sum

omega = np.linspace(0.0001, (np.pi/2))
IB1.set_xscale('log')
IB1.set_yscale('log')
IB1.set_xlim(10 ** (-2), (np.pi/2) * 10 ** 0)
IB1.set_ylim(10 ** (-4), 10 ** 2)
IB1.set_ylabel(r'$S_{PSD}(\omega)$')
IB1.plot(omega, RTS_PSD(omega))
IB1.plot(omega, (1/omega**1))



GF_group_STOPS_at_dT = []
GF_group_TEMPS_at_stop = []
n_GF=4
fig.set_tight_layout(True)
all_stop_durations = []
mean_delta_stop_temp = []
all_walk_durations = []
mean_delta_walk_temp = []
prob_dataset = []

for item in range(0,4,1):#itemizeGF:
    calc_dur_stop(Well_Behaved[n_GF])
    probability_T(Well_Behaved[n_GF])
    calc_dur_walk(Well_Behaved[n_GF])
    n_GF += 1

mean_t1 = np.mean(all_walk_durations)
start = 0
end = 6
dT = 0.1
v_GF = 1.475
bins = np.arange(start, end, dT)
rangeT = np.linspace(start, end, 59)
#nr_bins = 61
#bins = np.linspace(start, end, nr_bins)
data_entries_GF, bins_GF = np.histogram(prob_dataset, bins=bins, range=(start,end), density=True)
#GF1.hist(prob_dataset, bins, density=True)
#GF1.plot(rangeT, data_entries_GF)
GF1.annotate(r'$GF_{1-4}$', xy=(0.73,0.9),xycoords='axes fraction', fontsize=12)

#popt_stop1, pcov_stop1 = curve_fit(myCurve1, mean_delta_stop_temp, all_stop_durations, maxfev = 200000, p0=(0,0))#, bounds=([0,0], [3.0, 1.0]))
popt_stop2, pcov_stop2 = curve_fit(myCurve2, mean_delta_stop_temp, all_stop_durations, maxfev = 200000, p0=(0,0))#, bounds=([0,0], [3.0, 1.0]))
#popt_stop3, pcov_stop3 = curve_fit(myCurve3, mean_delta_stop_temp, all_stop_durations, maxfev = 200000, p0=(0,0,0))#, bounds=([0,0,0], [3.0, 1.0, 1.0]))
#xspace = np.linspace(0, 10, 1000)
#GF1.set_ylim(-10,1300)
#GF1.set_xlim(-0.01,8.01)
#GF1.plot(xspace, myCurve1(xspace, *popt_stop1), color='red', linewidth=1.1, linestyle='dashed', label=r'$ \tau / (1 + (\beta * \Delta T)) $')
#GF1.plot(xspace, myCurve2(xspace, *popt_stop2), color='red', linewidth=1.1, linestyle='dashdot', label=r'$ \tau * \exp(-\beta * \Delta T) $')
#GF1.plot(xspace, myCurve3(xspace, *popt_stop3), color='firebrick', linewidth=1.1, label=r'$ \tau * (\beta * \Delta T) ** \gamma $')
#GF1.scatter(mean_delta_stop_temp, all_stop_durations, color="black", alpha=0.3)
#GF1.set_ylabel(r'$t_{s}(s)$')

def RTS_PSD(w):#, v_0, P, t_0, t_1, range, dT):
    sum = 0
    for T in np.arange(0,5.9,0.1):
        sum += (data_entries_GF[int(round(T/0.1, 0))] * dT) / ((myCurve2(T, *popt_stop2) + 1) * ((1 / myCurve2(T, *popt_stop2) + 1 / mean_t1) ** 2 + w ** 2))
    return v_GF ** 2 * sum

omega = np.linspace(0.0001, (np.pi/2))
GF1.set_xscale('log')
GF1.set_yscale('log')
GF1.set_xlim(10 ** (-2), (np.pi/2) * 10 ** 0)
GF1.set_ylim(10 ** (-4), 10 ** 2)
GF1.plot(omega, RTS_PSD(omega))
GF1.plot(omega, (1/omega**1))


WF_group_STOPS_at_dT = []
WF_group_TEMPS_at_stop = []
n_WF=8
fig.set_tight_layout(True)
all_stop_durations = []
mean_delta_stop_temp = []
all_walk_durations = []
mean_delta_walk_temp = []
prob_dataset = []

for item in range(0,4,1):#itemizeWF:
    calc_dur_stop(Well_Behaved[n_WF])
    probability_T(Well_Behaved[n_WF])
    calc_dur_walk(Well_Behaved[n_WF])
    n_WF += 1

mean_t1 = np.mean(all_walk_durations)
start = 0
end = 6
dT = 0.1
v_WF = 2.097
bins = np.arange(start, end, dT)
rangeT = np.linspace(start, end, 59)
#nr_bins = 61
#bins = np.linspace(start, end, nr_bins)
data_entries_WF, bins_WF = np.histogram(prob_dataset, bins=bins, range=(start,end), density=True)
#WF1.hist(prob_dataset, bins, density=True)
#WF1.plot(rangeT, data_entries_WF)
WF1.annotate(r'$WF_{1-4}$', xy=(0.73,0.9),xycoords='axes fraction', fontsize=12)

#popt_stop1, pcov_stop1 = curve_fit(myCurve1, mean_delta_stop_temp, all_stop_durations, maxfev = 200000, p0=(0,0))#, bounds=([0,0], [3.0, 1.0]))
popt_stop2, pcov_stop2 = curve_fit(myCurve2, mean_delta_stop_temp, all_stop_durations, maxfev = 200000, p0=(0,0))#, bounds=([0,0], [3.0, 1.0]))
#popt_stop3, pcov_stop3 = curve_fit(myCurve3, mean_delta_stop_temp, all_stop_durations, maxfev = 200000, p0=(0,0,0))#, bounds=([0,0,0], [3.0, 1.0, 1.0]))
#xspace = np.linspace(0, 10, 1000)
#WF1.set_ylim(-10,1300)
#WF1.set_xlim(-0.01,8.01)
#WF1.plot(xspace, myCurve1(xspace, *popt_stop1), color='red', linewidth=1.1, linestyle='dashed', label=r'$ \tau / (1 + (\beta * \Delta T)) $')
#WF1.plot(xspace, myCurve2(xspace, *popt_stop2), color='red', linewidth=1.1, linestyle='dashdot', label=r'$ \tau * \exp(-\beta * \Delta T) $')
#WF1.plot(xspace, myCurve3(xspace, *popt_stop3), color='firebrick', linewidth=1.1, label=r'$ \tau * (\beta * \Delta T) ** \gamma $')
#WF1.scatter(mean_delta_stop_temp, all_stop_durations, color="black", alpha=0.3)
#WF1.set_ylabel(r'$t_{s}(s)$')

def RTS_PSD(w):#, v_0, P, t_0, t_1, range, dT):
    sum = 0
    for T in np.arange(0,5.9,0.1):
        sum += (data_entries_WF[int(round(T/0.1, 0))] * dT) / ((myCurve2(T, *popt_stop2) + 1) * ((1 / myCurve2(T, *popt_stop2) + 1 / mean_t1) ** 2 + w ** 2))
    return v_WF ** 2 * sum

omega = np.linspace(0.0001, (np.pi/2))
WF1.set_xscale('log')
WF1.set_yscale('log')
WF1.set_xlim(10 ** (-2), (np.pi/2) * 10 ** 0)
WF1.set_ylim(10 ** (-4), 10 ** 2)
WF1.set_xlabel(r'$\omega$')
WF1.set_ylabel(r'$S_{PSD}(\omega)$')
WF1.plot(omega, RTS_PSD(omega))
WF1.plot(omega, (1/omega**1))

RW_group_STOPS_at_dT = []
RW_group_TEMPS_at_stop = []
n_RW=12
fig.set_tight_layout(True)
all_stop_durations = []
mean_delta_stop_temp = []
all_walk_durations = []
mean_delta_walk_temp = []
prob_dataset = []

for item in range(0,4,1):#itemizeRW:
    calc_dur_stop(Well_Behaved[n_RW])
    probability_T(Well_Behaved[n_RW])
    calc_dur_walk(Well_Behaved[n_RW])
    n_RW += 1

mean_t1 = np.mean(all_walk_durations)
start = 0
end = 6
dT = 0.1
v_RW = 2.491
bins = np.arange(start, end, dT)
rangeT = np.linspace(start, end, 59)
#nr_bins = 61
#bins = np.linspace(start, end, nr_bins)
data_entries_RW, bins_RW = np.histogram(prob_dataset, bins=bins, range=(start,end), density=True)
#RW1.hist(prob_dataset, bins, density=True)
#RW1.plot(rangeT, data_entries_RW)
RW1.annotate(r'$RW_{1-4}$', xy=(0.73,0.9),xycoords='axes fraction', fontsize=12)

#popt_stop1, pcov_stop1 = curve_fit(myCurve1, mean_delta_stop_temp, all_stop_durations, maxfev = 200000, p0=(0,0))#, bounds=([0,0], [3.0, 1.0]))
popt_stop2, pcov_stop2 = curve_fit(myCurve2, mean_delta_stop_temp, all_stop_durations, maxfev = 200000, p0=(0,0))#, bounds=([0,0], [3.0, 1.0]))
#popt_stop3, pcov_stop3 = curve_fit(myCurve3, mean_delta_stop_temp, all_stop_durations, maxfev = 200000, p0=(0,0,0))#, bounds=([0,0,0], [3.0, 1.0, 1.0]))
#xspace = np.linspace(0, 10, 1000)
#RW1.set_ylim(-10,1300)
#RW1.set_xlim(-0.01,8.01)
#RW1.plot(xspace, myCurve1(xspace, *popt_stop1), color='red', linewidth=1.1, linestyle='dashed', label=r'$ \tau / (1 + (\beta * \Delta T)) $')
#RW1.plot(xspace, myCurve2(xspace, *popt_stop2), color='red', linewidth=1.1, linestyle='dashdot', label=r'$ \tau * \exp(-\beta * \Delta T) $')
#RW1.plot(xspace, myCurve3(xspace, *popt_stop3), color='firebrick', linewidth=1.1, label=r'$ \tau * (\beta * \Delta T) ** \gamma $')
#RW1.scatter(mean_delta_stop_temp, all_stop_durations, color="black", alpha=0.3)
#RW1.set_ylabel(r'$t_{s}(s)$')

def RTS_PSD(w):#, v_0, P, t_0, t_1, range, dT):
    sum = 0
    for T in np.arange(0,5.9,0.1):
        sum += (data_entries_RW[int(round(T/0.1, 0))] * dT) / ((myCurve2(T, *popt_stop2) + 1) * ((1 / myCurve2(T, *popt_stop2) + 1 / mean_t1) ** 2 + w ** 2))
    return v_RW ** 2 * sum

omega = np.linspace(0.0001, (np.pi/2))
RW1.set_xscale('log')
RW1.set_yscale('log')
RW1.set_xlim(10 ** (-2), (np.pi/2) * 10 ** 0)
RW1.set_ylim(10 ** (-4), 10 ** 2)
RW1.set_xlabel(r'$\omega$')
RW1.plot(omega, RTS_PSD(omega))
RW1.plot(omega, (1/omega**1), label=r'$1 / \omega^{1}$')
RW1.legend(loc="lower left")
plt.show()
