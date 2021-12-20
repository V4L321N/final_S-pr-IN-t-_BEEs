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

# bad_bees = ["BT11A-1","BT11A-4", "BT01A-1","BT02B-1", "BT03A-1", "BT03A-2", "BT03A-3", "BT03A-4", "BT03B-1", "BT03B-2", "BT03B-3", "BT03B-4", "BT07A-1", "BT07A-2", "BT07A-3", "BT07A-4", "BT07B-1", "BT07B-2", "BT07B-3", "BT07B-4", "BT17A-1", "BT17A-2", "BT17A-3", "BT17A-4", "BT17B-1", "BT17B-2", "BT17B-3", "BT17B-4"]
# #BT11A-1 and "BT11A-4 needs to be removed from the bad bees list.
# for item in bad_bees:
#     head.remove(item)

#testbee = "BT01A-1"
def arena_temperature(item):
    return pd.DataFrame(pd.read_csv("data_bee_types/temperature_variables.csv"), columns=[item])

def manual_type(item):
    return  pd.DataFrame(pd.read_csv("data_bee_types/bee_types_manual.csv"), columns=[item])

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


def S(w, D, coupling_a):
    return 2 * D / (coupling_a ** 2 + w ** 2)


ALL_a_list = []
ALL_D_r_list = []

for testbee in head:
    psd_test = PSD(testbee)
    length = len(psd_test)
    start = 0.001
    end = 2*np.pi/2
    omega = np.linspace(start, end, length)
    popt, pcov = curve_fit(S, omega, psd_test)
    ALL_D_r_list.append(popt[0])
    ALL_a_list.append(popt[1])
    #print(grouped_bee, "D_r= " + str(popt[0]), "a= " + str(popt[1]))


# plt.hist(ALL_D_r_list, bins=40)#, range=(0,2))
# plt.xlabel("D_r")
# plt.ylabel("occurence")
# plt.title("ALL BEES")
# plt.savefig("FIG_hist_per_type/ALL_D_r")
# plt.close()
# plt.show()
#
# plt.hist(ALL_a_list, bins=40)#, range=(0,0.2))
# plt.xlabel("a")
# plt.ylabel("occurence")
# plt.title("ALL BEES")
# plt.savefig("FIG_hist_per_type/ALL_a")
# plt.close()



plt.scatter(ALL_D_r_list, ALL_a_list, color='black', alpha=0.5)
x = ALL_D_r_list
y = ALL_a_list
xaxis = np.linspace(min(x), max(x))
m, b = np.polyfit(x,y,1)
plt.plot(xaxis, m*xaxis + b, color='black', linestyle='dotted')
plt.annotate(r'$m_{\theta}=$'+str(round(m,3)), xy=(0.05,0.11),xycoords='axes fraction', fontsize=10)
plt.annotate(r'$b_{\theta}=$'+str(round(b,3)), xy=(0.05,0.05),xycoords='axes fraction', fontsize=10)
plt.ylabel(r'$a_{\theta}$')
plt.xlabel(r'$D_{\theta}$')
plt.show()
#plt.title("ALL BEES")
#plt.savefig("FIG_hist_per_type/ALL_a_VS_D_r")
