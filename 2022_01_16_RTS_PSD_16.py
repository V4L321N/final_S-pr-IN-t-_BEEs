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
    prob_dataset = []
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
    all_stop_durations = []
    stop_temperatures = []
    mean_delta_stop_temp = []
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
"""-end--------calculate duration walking-----------------"""



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

"""-end--------calculate velocity  (except zero) in for each time step-----------------"""

"""-begin------calculate velocity for each time step-----------------"""
def calc_vel_new(item):
    list_vel = []
    X_dataset_wo_NAN = X_remove_NaN(item) #test_x
    Y_dataset_wo_NAN = Y_remove_NaN(item) #test_y
    n = len(X_dataset_wo_NAN) - 1
    timeline = np.linspace(0, n, n)
    for t in range(n):
        velocity = np.sqrt((X_dataset_wo_NAN[t + 1] - X_dataset_wo_NAN[t]) ** 2 + (Y_dataset_wo_NAN[t + 1] - Y_dataset_wo_NAN[t]) ** 2)
        list_vel.append(velocity)
    return list_vel
"""-end--------calculate velocity in for each time step-----------------"""


"""-begin--------calculate fourier transform of the velocity-----------------"""
def velocity_FFT(item):
    return np.fft.fft(calc_vel_new(item), norm="ortho") #either norm="ortho" here or divided by C_2 in PSD()
"""-end--------calculate fourier transform of the velocity-------------------"""


"""-begin--------calculate power spectral density of the FT------------------"""
def PSD(item):
    vel_FFT = velocity_FFT(item)
    loop_L = round(len(vel_FFT)/2)
    PSD_list = []
    for i in range(loop_L):
        C_2 = 1
        absolute = C_2 * (vel_FFT[i] * np.conj(vel_FFT[i]))
        PSD_list.append(absolute)
        #PSD_list.append(np..sqrt(absolute.real ** 2 + absolute.imag ** 2))
    return PSD_list

# def PSD(item):
#     PSD_list = []
#     for i in range(round(len(theta_FFT)/2)):
#         PSD_list.append(theta_FFT[i].real ** 2 + theta_FFT[i].imag ** 2)
#     return PSD_list
"""-end----------calculate power spectral density of the angle FT------------"""


"""-begin--------fitting functions------------------"""
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

def powerCurve(f, alpha):
    return 1/(f**alpha)

"""-end--------fitting functions------------------"""

fig, ((IB1, IB2, IB3, IB4), (GF1, GF2, GF3, GF4), (WF1, WF2, WF3, WF4), (RW1, RW2, RW3, RW4)) = plt.subplots(4, 4, figsize=(10,10))
fig.set_tight_layout(True)
#fig, ((IB1,GF1),(WF1,RW1)) = plt.subplots(2, 2, figsize=(7,7))
itemize = [IB1, IB2, IB3, IB4, GF1, GF2, GF3, GF4, WF1, WF2, WF3, WF4, RW1, RW2, RW3, RW4]
type = [r'$IB_1$', r'$IB_2$', r'$IB_3$', r'$IB_4$', r'$GF_1$', r'$GF_2$', r'$GF_3$', r'$GF_4$', r'$WF_1$', r'$WF_2$', r'$WF_3$', r'$WF_4$', r'$RW_1$', r'$RW_2$', r'$RW_3$', r'$RW_4$']
n = 0
start = 0
end = 6
dT = 0.1
xspace = np.linspace(0, 10, 1000)
for item in itemize:
    item.annotate(type[n], xy=(0.8,0.9),xycoords='axes fraction', fontsize=12)
    if item != IB2:
        vel_PSD = PSD(Well_Behaved[n])
        length = len(vel_PSD)
        start_new = 0.001
        end_new = 2 * np.pi / 2
        omega_new = np.linspace(start_new, end_new, length)
        item.plot(omega_new, vel_PSD, color='black', alpha=0.5)

        mean_velocity = np.mean(calc_vel(Well_Behaved[n])[0])
        all_stop_durations, mean_delta_stop_temp = calc_dur_stop(Well_Behaved[n])
        all_walk_durations, mean_delta_walk_temp = calc_dur_walk(Well_Behaved[n])
        mean_t1 = np.mean(all_walk_durations)
        #popt_stop1, pcov_stop1 = curve_fit(myCurve1, mean_delta_stop_temp, all_stop_durations, maxfev = 200000, p0=(0,0))#, bounds=([0,0], [3.0, 1.0]))
        popt_stop2, pcov_stop2 = curve_fit(myCurve2, mean_delta_stop_temp, all_stop_durations, maxfev = 200000, p0=(0,0))#, bounds=([0,0], [3.0, 1.0]))
        #popt_stop3, pcov_stop3 = curve_fit(myCurve3, mean_delta_stop_temp, all_stop_durations, maxfev = 200000, p0=(0,0,0))#, bounds=([0,0,0], [3.0, 1.0, 1.0]))

        prob_dataset = probability_T(Well_Behaved[n])
        bins = np.arange(start, end, dT)
        rangeT = np.linspace(start, end, 59)
        data_entries, bins = np.histogram(prob_dataset, bins=bins, range=(start,end), density=True)

        def S(w, v_0, t_1, Prob, dTemp):
            sum = 0
            for T in np.arange(0,5.9,0.1):
                sum += (Prob[int(round(T/0.1, 0))] * dTemp) / ((myCurve2(T, *popt_stop2) + 1) * ((1 / myCurve2(T, *popt_stop2) + 1 / t_1) ** 2 + w ** 2))
            return 2 * v_0 ** 2 * sum
            #    sum_other += (Prob[int(round(T/0.1, 0))] * dTemp * (1 / myCurve2(T, *popt_stop2) + 1 / t_1)) / ((1 / myCurve2(T, *popt_stop2) + 1 / t_1) ** 2 + w ** 2)
            #return v_0 ** 2 * sum_other
        omega = np.linspace(0.0001, (np.pi/2))
        item.plot(omega, S(omega, mean_velocity, mean_t1, data_entries, dT), linestyle='dashed', color='black')


        x_low = 0.1
        x_high = 1.0
        y_low = S(x_low, mean_velocity, mean_t1, data_entries, dT)
        y_high = S(x_high, mean_velocity, mean_t1, data_entries, dT)
        item.plot(0.1, y_low , marker="^", color='red')
        item.plot(1, y_high, marker="^", color='red')
        #item.plot(omega, y_low*omega/omega, linestyle='dotted', color='red', linewidth=0.5)
        #item.plot(omega, y_high*omega/omega, linestyle='dotted', color='red', linewidth=0.5)
        alpha = np.abs(np.log10(y_high/y_low))
        item.plot(omega, powerCurve(omega, alpha), linestyle='dotted', color='red')
        item.annotate(r'$\alpha = $' + str(round(alpha,2)) , xy=(0.05,0.9),xycoords='axes fraction', fontsize=10)

        #high_omega = np.linspace(0.2,(np.pi/2))
        #popt_pow, pcov_pow = curve_fit(powerCurve, high_omega, S(high_omega, mean_velocity, mean_t1, data_entries, dT), maxfev=200000, p0=(2))
        #item.plot(omega, powerCurve(high_omega, *popt_pow), linestyle='dotted', color='red')
        #item.annotate(r'$\alpha = $' + str(round(popt_pow[0],2)) , xy=(0.1,0.9),xycoords='axes fraction', fontsize=12)
    else:
        item.annotate('invalid', xy=(0.35,0.45),xycoords='axes fraction', color='red', fontsize=12)

    item.set_xscale('log')
    item.set_yscale('log')
    item.set_xlim(10 ** (-3), (np.pi/2) * 10 ** 0)
    item.set_ylim(10 ** (-4), 10 ** 4)
    #item.set_xlabel()
    #item.set_ylabel()
    item.set_xticks([])
    item.set_yticks([])
    if item in [IB1, GF1, WF1, RW1]:
        item.set_ylabel(r'$S_{RTS}(\omega)$')
        item.set_yticks([10 ** (-2), 10**(0), 10 ** (2)])
    if item in [RW1, RW2, RW3, RW4]:
        item.set_xlabel(r'$\omega$')
        item.set_xticks([10 ** (-2), 10**(-1), 10 ** 0])


    n += 1
plt.show()



#nr_bins = 61
#bins = np.linspace(start, end, nr_bins)

#RW1.hist(prob_dataset, bins, density=True)
#RW1.plot(rangeT, data_entries_RW)



#xspace = np.linspace(0, 10, 1000)
#RW1.set_ylim(-10,1300)
#RW1.set_xlim(-0.01,8.01)
#RW1.plot(xspace, myCurve1(xspace, *popt_stop1), color='red', linewidth=1.1, linestyle='dashed', label=r'$ \tau / (1 + (\beta * \Delta T)) $')
#RW1.plot(xspace, myCurve2(xspace, *popt_stop2), color='red', linewidth=1.1, linestyle='dashdot', label=r'$ \tau * \exp(-\beta * \Delta T) $')
#RW1.plot(xspace, myCurve3(xspace, *popt_stop3), color='firebrick', linewidth=1.1, label=r'$ \tau * (\beta * \Delta T) ** \gamma $')
#RW1.scatter(mean_delta_stop_temp, all_stop_durations, color="black", alpha=0.3)
#RW1.set_ylabel(r'$t_{s}(s)$')
