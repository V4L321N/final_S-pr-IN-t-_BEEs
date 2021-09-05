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


for testbee in head:
    plt.title(testbee)
    plt.xlim(0, 6, 0.1)
    #plt.ylim(0, 10)
    plt.xlabel('velocity in cm/s')
    plt.ylabel('counts')
    #plt.yscale('log')
    plt.hist(calc_vel(testbee)[0], bins=30, range=(0,6), density=True, alpha=0.50, label='valid data')
    plt.hist(calc_vel(testbee)[1], bins=30, range=(0,6), density=True, alpha=0.50, color='red', label='removed data')
    std_walk = np.std(calc_vel(testbee)[0], ddof=1)
    mean_walk = np.mean(calc_vel(testbee)[0])
    std_stop = np.std(calc_vel(testbee)[1], ddof=1)
    mean_stop = 0 #np.mean(calc_vel(testbee)[1])
    domain = np.linspace(0, 6)
    plt.plot(domain, norm.pdf(domain, mean_walk, std_walk), color='black', linestyle='dashed', label = '$\mathcal{N}$ ' + f'$( \mu \\approx {round(mean_walk, 4)}, \sigma \\approx {round(std_walk, 4)} )$')
    plt.plot(domain, norm.pdf(domain, mean_stop, std_stop), color='red', linestyle='dashed', label = '$\mathcal{N}$ ' + f'$( \mu \\approx {round(mean_stop, 4)}, \sigma \\approx {round(std_stop, 4)} )$')
    plt.legend()
    #plt.savefig("FIG_FIT_of_HIST/FIT_v0_" + str(testbee))
    #plt.close()

    plt.show()
