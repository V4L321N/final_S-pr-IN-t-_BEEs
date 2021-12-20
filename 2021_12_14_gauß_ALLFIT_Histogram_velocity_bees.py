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
"""-end--------remove empty entires------------------------------------------"""


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

# def PSD(item):
#     PSD_list = []
#     for i in range(round(len(theta_FFT)/2)):
#         PSD_list.append(theta_FFT[i].real ** 2 + theta_FFT[i].imag ** 2)
#     return PSD_list
"""-end----------calculate power spectral density of the angle FT------------"""
list_vel = []
list_vel_NO_stopping = []
list_vel_ONLY_stopping = []
for item in head:
    histogram_data = calc_vel(item)


plt.hist(histogram_data[0], bins=30, range=(0,6), density=True, alpha=0.50, color='blue', label='valid data')
plt.hist(histogram_data[1], bins=30, range=(0,6), density=True, alpha=0.50, color='red', label='removed data')
std_walk = np.std(histogram_data[0], ddof=1)
mean_walk = np.mean(histogram_data[0])
domain = np.linspace(0, 6)
plt.ylim(0,2)
plt.plot(domain, norm.pdf(domain, mean_walk, std_walk), color='black', linestyle='dashed')
plt.show()
# fig, ((IB1, IB2, IB3, IB4), (GF1, GF2, GF3, GF4), (WF1, WF2, WF3, WF4), (RW1, RW2, RW3, RW4)) = plt.subplots(4, 4, figsize=(10,10))
# itemize = [IB1, IB2, IB3, IB4, GF1, GF2, GF3, GF4, WF1, WF2, WF3, WF4, RW1, RW2, RW3, RW4]
# type = [r'$IB_1$', r'$IB_2$', r'$IB_3$', r'$IB_4$', r'$GF_1$', r'$GF_2$', r'$GF_3$', r'$GF_4$', r'$WF_1$', r'$WF_2$', r'$WF_3$', r'$WF_4$', r'$RW_1$', r'$RW_2$', r'$RW_3$', r'$RW_4$']
# n=0
# fig.set_tight_layout(True)
# for item in itemize:
#     item.set_xlim(0, 6.1, 0.1)
#     item.set_ylim(0,5.1)
#     item.hist(calc_vel(Well_Behaved[n])[0], bins=30, range=(0,6), density=True, alpha=0.50, color='blue', label='valid data')
#     item.hist(calc_vel(Well_Behaved[n])[1], bins=30, range=(0,6), density=True, alpha=0.50, color='red', label='removed data')
#     std_walk = np.std(calc_vel(Well_Behaved[n])[0], ddof=1)
#     mean_walk = np.mean(calc_vel(Well_Behaved[n])[0])
#     #std_stop = np.std(calc_vel(Well_Behaved[n])[1], ddof=1)
#     #mean_stop = 0 #np.mean(calc_vel(testbee)[1])
#     domain = np.linspace(0, 6)
#     item.set_xticks([])
#     item.set_yticks([])
#     item.annotate(type[n], xy=(0.8,0.9),xycoords='axes fraction', fontsize=12)
#     item.plot(domain, norm.pdf(domain, mean_walk, std_walk), color='black', linestyle='dashed')
#     #item.plot(domain, norm.pdf(domain, mean_stop, std_stop), color='red', linestyle='dashed')
#     item.annotate(r'$\mu \approx $'+ str(round(mean_walk, 3)) + ' ' + r'$cm/s$', xy=(0.1,0.9), xycoords='axes fraction', fontsize=10)
#     item.annotate(r'$\sigma \approx $'+ str(round(std_walk, 3)) + ' ' + r'$cm/s$', xy=(0.1,0.8), xycoords='axes fraction', fontsize=10)
#     if n in [0,4,8,12]:
#         item.set_ylabel(r'$counts$')
#         item.set_yticks([0,5])
#     if n in [12,13,14,15]:
#         item.set_xlabel('velocity ' + r'$(cm/s)$')
#         item.set_xticks([0, 2, 4, 6])
#     #item.legend()
#     n += 1
#     #plt.savefig("FIG_FIT_of_HIST/FIT_v0_" + str(testbee))
#     #plt.close()
# plt.show()
