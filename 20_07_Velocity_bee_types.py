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

#test_x = [60, 50, 40, 30, 20, 10, 0, 10, 20, 30, 40, 50, 60, 50, 40, 30, 20, 10, 0, 10, 20, 30, 40, 50, 60, 50, 40, 30, 20, 10, 0, 10, 20, 30, 40, 50, 60]
#test_y = [30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30]
#testbee = "BT01A-1"

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

def S(w, D, coupling_a):
    return 2 * D / (coupling_a ** 2 + w ** 2)

D_v_list = []
a_list = []
testbee_list = []

for testbee in head:
    psd_test = PSD(testbee)
    length = len(psd_test)
    start = 0.001
    end = 2 * np.pi * 2
    omega = np.linspace(start, end, length)
    popt, pcov = curve_fit(S, omega, psd_test)#, bounds=([0, 0], [10000, 10]))
    D_v_list.append(popt[0])
    a_list.append(popt[1])
    testbee_list.append(testbee)
    print(testbee, "D_v= " + str(popt[0]), "a= " + str(popt[1]))
    plt.plot(omega, PSD(testbee))
    plt.plot(omega, S(omega, popt[0], popt[1]))
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(start, end)
    plt.xlabel('\u03C9')
    plt.ylabel('S[v]')
    plt.title(testbee)

    plt.savefig("FIG_D_v/" + str(testbee))

    plt.close()
    #plt.show()


# testbee = "BT15A-1"
# psd_test = PSD(testbee)
# length = len(psd_test)
# omega = np.linspace(0, 100, length)
# popt, pcov = curve_fit(S, omega, psd_test, bounds=([0, 0], [10000, 10]))
# D_v_list.append(popt[0])
# a_list.append(popt[1])
# testbee_list.append(testbee)
# print(testbee, "D_v= " + str(popt[0]), "a= " + str(popt[1]))
# plt.plot(omega, PSD(testbee))
# plt.plot(omega, S(omega, popt[0], popt[1]))
# plt.xscale('log')
# plt.yscale('log')
# plt.xlim(0, 100)
# plt.xlabel('\u03C9')
# plt.ylabel('S[v]')
# plt.title(testbee)
# plt.show()



"""-manual tryout for fit----------------------------------------------------"""
# def generate_S(w, D, coupling_a):
#     S_list = []
#     for i in range(len(w)):
#         S_list.append(2 * D / (coupling_a ** 2 + w[i] ** 2))
#     return S_list
#plt.plot(omega, generate_S(omega, D_r, a))

# def residual_squares(func_1, func_2):
#     residuals = []
#     res_sum = 0
#     for i in range(len(func_1)):
#         residual_i = (func_1[i] - func_2[i]) ** 2
#         res_sum += residual_i
#         residuals.append(residual_i)
#     return res_sum #residuals

# fit = []
# for i in range(10):
#     fit.append(residual_squares(PSD(testbee), generate_S(omega, D_r, a)))
#     D_r += 100
#     a += 0.1
# plt.plot(fit)
