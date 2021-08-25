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

head = list(pd.DataFrame(data=pd.read_csv("data_bee_types/temperature_variables.csv")))

#testbee = "BT01A-1"
def arena_temperature(item):
    return pd.DataFrame(pd.read_csv("data_bee_types/temperature_variables.csv"), columns=[item])

def manual_type(item):
    return  pd.DataFrame(pd.read_csv("data_bee_types/bee_types_manual.csv"), columns=[item])

grad_NO_delta = []
grad_TWO_delta = []
grad_FOUR_delta = []
grad_SIX_delta = []
grad_EIGHT_delta = []
grad_TEN_delta = []

for testbee in head:
    arena_temp = arena_temperature(testbee)
    manual_t = manual_type(testbee)
    # upper = round(arena_temp.iloc[1].item())
    # lower = round(arena_temp.iloc[0].item())
    # DELTA = upper - lower
    upper = arena_temp.iloc[1].item()
    lower = arena_temp.iloc[0].item()
    type = manual_t.iloc[0].item()
    DELTA = round(upper-lower)
    if DELTA < 1:
        grad_NO_delta.append([testbee, type])
    elif 2 >= DELTA >= 1:
        grad_TWO_delta.append([testbee, type])
    elif 4 >= DELTA > 2:
        grad_FOUR_delta.append([testbee, type])
    elif 6 >= DELTA > 4:
        grad_SIX_delta.append([testbee, type])
    elif 8 >= DELTA > 6:
        grad_EIGHT_delta.append([testbee, type])
    elif DELTA > 8:
        grad_TEN_delta.append([testbee, type])

# print(len(grad_NO_delta))
# print(len(grad_TWO_delta))
# print(len(grad_FOUR_delta))
# print(len(grad_SIX_delta))
# print(len(grad_EIGHT_delta))
# print(len(grad_TEN_delta))

# for testbee in head:
#     arena_temp = arena_temperature(testbee)
#     manual_t = manual_type(testbee)
#     # upper = round(arena_temp.iloc[1].item())
#     # lower = round(arena_temp.iloc[0].item())
#     # DELTA = upper - lower
#     upper = arena_temp.iloc[1].item()
#     lower = arena_temp.iloc[0].item()
#     type = manual_t.iloc[0].item()
#     DELTA = round(upper-lower)
#     if DELTA < 1:
#         grad_NO_delta.append(testbee)
#     elif 2 >= DELTA >= 1:
#         grad_TWO_delta.append(testbee)
#     elif 4 >= DELTA > 2:
#         grad_FOUR_delta.append(testbee)
#     elif 6 >= DELTA > 4:
#         grad_SIX_delta.append(testbee)
#     elif 8 >= DELTA > 6:
#         grad_EIGHT_delta.append(testbee)
#     elif DELTA > 8:
#         grad_TEN_delta.append(testbee)

_0D_IBs = []
_0D_RWs = []
_0D_WFs = []
_0D_GFs = []

for grad_testbee in grad_NO_delta:
    if grad_testbee[1] == "Immobile Bee":
        _0D_IBs.append(grad_testbee[0])
    elif grad_testbee[1] == "Random Walker":
        _0D_RWs.append(grad_testbee[0])
    elif grad_testbee[1] == "Goal Finder":
        _0D_GFs.append(grad_testbee[0])
    elif grad_testbee[1] == "Wall Follower":
        _0D_WFs.append(grad_testbee[0])

# print(len(_0D_IBs))
# print(len(_0D_RWs))
# print(len(_0D_WFs))
# print(len(_0D_GFs))


_2D_IBs = []
_2D_RWs = []
_2D_WFs = []
_2D_GFs = []

for grad_testbee in grad_TWO_delta:
    if grad_testbee[1] == "Immobile Bee":
        _2D_IBs.append(grad_testbee[0])
    elif grad_testbee[1] == "Random Walker":
        _2D_RWs.append(grad_testbee[0])
    elif grad_testbee[1] == "Goal Finder":
        _2D_GFs.append(grad_testbee[0])
    elif grad_testbee[1] == "Wall Follower":
        _2D_WFs.append(grad_testbee[0])

# print(len(_2D_IBs))
# print(len(_2D_RWs))
# print(len(_2D_WFs))
# print(len(_2D_GFs))


_4D_IBs = []
_4D_RWs = []
_4D_WFs = []
_4D_GFs = []

for grad_testbee in grad_FOUR_delta:
    if grad_testbee[1] == "Immobile Bee":
        _4D_IBs.append(grad_testbee[0])
    elif grad_testbee[1] == "Random Walker":
        _4D_RWs.append(grad_testbee[0])
    elif grad_testbee[1] == "Goal Finder":
        _4D_GFs.append(grad_testbee[0])
    elif grad_testbee[1] == "Wall Follower":
        _4D_WFs.append(grad_testbee[0])

# print(len(_4D_IBs))
# print(len(_4D_RWs))
# print(len(_4D_WFs))
# print(len(_4D_GFs))


_6D_IBs = []
_6D_RWs = []
_6D_WFs = []
_6D_GFs = []

for grad_testbee in grad_SIX_delta:
    if grad_testbee[1] == "Immobile Bee":
        _6D_IBs.append(grad_testbee[0])
    elif grad_testbee[1] == "Random Walker":
        _6D_RWs.append(grad_testbee[0])
    elif grad_testbee[1] == "Goal Finder":
        _6D_GFs.append(grad_testbee[0])
    elif grad_testbee[1] == "Wall Follower":
        _6D_WFs.append(grad_testbee[0])

# print(len(_6D_IBs))
# print(len(_6D_RWs))
# print(len(_6D_WFs))
# print(len(_6D_GFs))


_8D_IBs = []
_8D_RWs = []
_8D_WFs = []
_8D_GFs = []

for grad_testbee in grad_EIGHT_delta:
    if grad_testbee[1] == "Immobile Bee":
        _8D_IBs.append(grad_testbee[0])
    elif grad_testbee[1] == "Random Walker":
        _8D_RWs.append(grad_testbee[0])
    elif grad_testbee[1] == "Goal Finder":
        _8D_GFs.append(grad_testbee[0])
    elif grad_testbee[1] == "Wall Follower":
        _8D_WFs.append(grad_testbee[0])

# print(len(_8D_IBs))
# print(len(_8D_RWs))
# print(len(_8D_WFs))
# print(len(_8D_GFs))


_10D_IBs = []
_10D_RWs = []
_10D_WFs = []
_10D_GFs = []

for grad_testbee in grad_TEN_delta:
    if grad_testbee[1] == "Immobile Bee":
        _10D_IBs.append(grad_testbee[0])
    elif grad_testbee[1] == "Random Walker":
        _10D_RWs.append(grad_testbee[0])
    elif grad_testbee[1] == "Goal Finder":
        _10D_GFs.append(grad_testbee[0])
    elif grad_testbee[1] == "Wall Follower":
        _10D_WFs.append(grad_testbee[0])

# print(len(_10D_IBs))
# print(len(_10D_RWs))
# print(len(_10D_WFs))
# print(len(_10D_GFs))


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

for grouped_bee in _2D_RWs:
    psd_test = PSD(grouped_bee)
    length = len(psd_test)
    #freq = np.linspace(0, 1, length)
    #fit_freq = np.linspace(0, 10, length)
    #omega = np.linspace(0.001, np.pi, length)
    start = 0.001
    end = 2*np.pi/2#0.01
    omega = np.linspace(start, end, length)
    #omega_fit = np.linspace(0, 1, length)
    popt, pcov = curve_fit(S, omega, psd_test)#, bounds=([0, 0], [10000, 10]), method='dogbox')
    D_r_list.append(popt[0])
    a_list.append(popt[1])
    print(grouped_bee, "D_r= " + str(popt[0]), "a= " + str(popt[1]))
    # plt.plot(omega, PSD(grouped_bee))
    # plt.plot(omega, S(omega, popt[0], popt[1]))
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.xlim(start, end)
    # plt.xlabel('f')
    # plt.ylabel('S[\u03B8]')
    # plt.title(grouped_bee)
    #plt.show()

plt.hist(D_r_list, bins=20)#, range=(0,2))
plt.show()
plt.hist(a_list, bins=20)#, range=(0,0.2))
plt.show()
