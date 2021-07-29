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


"""-begin------calculate turning angle in for each time step-----------------"""
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
#plt.plot(calc_theta(testbee))
"""-end--------calculate turning angle in for each time step-----------------"""


"""-begin------calculate position of the gradient depending on bee position--"""
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
#plt.plot(calc_dir_G(testbee))
"""-end--------calculate position of the gradient depending on bee position--"""


"""-begin------calculate turning angle in respect of gradient position-------"""
def theta_in_G(item):
    list_theta_in_G = []
    theta = calc_theta(item)
    dir_G = calc_dir_G(item)
    n = len(dir_G)
    for t in range(n):
        list_theta_in_G.append(theta[t] - dir_G[t])
    return list_theta_in_G
#plt.plot(theta_in_G(testbee))
"""-end--------calculate turning angle in respect of gradient position-------"""


"""-begin------recalculate turning angle to avoid jumps from -pi and +pi-----"""
def recalc_w_2pi(item):
    theta_G = theta_in_G(item)
    n = len(theta_G)
    for t in range(n):
        if theta_G[t] - theta_G[t-1] >= np.pi:
            theta_G[t] -= 2 * np.pi
        elif theta_G[t] - theta_G[t-1] <= -np.pi:
            theta_G[t] += 2 * np.pi
    return theta_G
#plt.plot(recalc_w_2pi(testbee))
"""-end--------recalculate turning angle to avoid jumps from -pi and +pi-----"""


#-begin--------calculate fourier transform of the turning angle-----------------
def theta_FFT(item):
    return np.fft.fft(recalc_w_2pi(item), norm="ortho") #either norm="ortho" here or divided by C_1 in PSD()
#-end--------calculate fourier transform of the turning angle-------------------


"""-begin--------calculate power spectral density of the FT------------------"""
def PSD(item):
    th_FFT = theta_FFT(item)
    loop_L = round(len(th_FFT)/2)
    PSD_list = []
    for i in range(loop_L):
        C_1 = 1#/len(theta_FFT)
        absolute = C_1 * (th_FFT[i] * np.conj(th_FFT[i]))
        PSD_list.append(absolute.real)
    return PSD_list

# def PSD(item):
#     PSD_list = []
#     for i in range(round(len(theta_FFT)/2)):
#         PSD_list.append(theta_FFT[i].real ** 2 + theta_FFT[i].imag ** 2)
#     return PSD_list
"""-end----------calculate power spectral density of the angle FT------------"""




# plt.xscale('log')
# plt.yscale('log')
# plt.plot(omega, PSD(testbee))
#S_theta = 2 * D_r / (a ** 2 + omega ** 2)
#plt.plot(omega, S_theta)

def S(w, D, coupling_a):
    return 2 * D / (coupling_a ** 2 + w ** 2)

a_list = []
D_r_list = []
testbee_list = []

for testbee in head:
    psd_test = PSD(testbee)
    length = len(psd_test)
    start = 0.001
    end = 2 * np.pi / 2
    #freq = np.linspace(0, 1, length)
    omega = np.linspace(start, end, length)
    popt, pcov = curve_fit(S, omega, psd_test)#, bounds=([0, 0], [10000, 10]))
    D_r_list.append(popt[0])
    a_list.append(popt[1])
    testbee_list.append(testbee)
    print(testbee, "D_r= " + str(popt[0]), "a= " + str(popt[1]))
    plt.plot(omega, PSD(testbee))
    plt.plot(omega, S(omega, popt[0], popt[1]))
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(start, end)
    plt.xlabel('omega')
    plt.ylabel('S[\u03B8]')
    plt.title(testbee)

    plt.savefig("FIG_D_r/" + str(testbee))

    plt.close()

    #plt.show()

# testbee = "BT12B-1"
# psd_test = PSD(testbee)
# length = len(psd_test)
# omega = np.linspace(0, 100, length)
# popt, pcov = curve_fit(S, omega, psd_test, bounds=([0, 0], [10000, 10]))
# D_r_list.append(popt[0])
# a_list.append(popt[1])
# testbee_list.append(testbee)
# print(testbee, "D_r= " + str(popt[0]), "a= " + str(popt[1]))
# plt.plot(omega, PSD(testbee))
# plt.plot(omega, S(omega, popt[0], popt[1]))
# plt.xscale('log')
# plt.yscale('log')
# plt.xlim(0, 100)
# plt.xlabel('\u03C9')
# plt.ylabel('S[\u03B8]')
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
