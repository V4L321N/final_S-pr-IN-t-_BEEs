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
#IB_1 = "BT01A-1"
IB_1 = "BT06A-2"
IB_2 = "BT06A-3"
IB_3 = "BT02B-2"
IB_4 = "BT02B-1"
GF_1 = "BT09A-2"
GF_2 = "BT09B-2"
#GF_3 = "BT10B-4" #new1
#GF_3 = "BT01C-4" #new1
GF_3 = "BT09B-4"
GF_4 = "BT12B-2"
WF_1 = "BT04B-3"
WF_2 = "BT09B-1"
WF_3 = "BT12B-1"
WF_4 = "BT13B-3"
#RW_1 = "BT18A-4" #new1
RW_1 = "BT03A-1"
RW_2 = "BT06B-1"
RW_3 = "BT13A-3"
RW_4 = "BT12A-1"
Well_Behaved = [IB_1, IB_2, IB_3, IB_4, GF_1, GF_2, GF_3, GF_4, WF_1, WF_2, WF_3, WF_4, RW_1, RW_2, RW_3, RW_4]

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
        while theta_G[t] - theta_G[t-1] > np.pi:
            theta_G[t] -= 2 * np.pi
        while theta_G[t] - theta_G[t-1] < -np.pi:
            theta_G[t] += 2 * np.pi
        # if theta_G[t] - theta_G[t-1] > np.pi:
        #     theta_G[t] -= 2 * np.pi
        # elif theta_G[t] - theta_G[t-1] < -np.pi:
        #     theta_G[t] += 2 * np.pi
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

fig, ((IB1, IB2, IB3, IB4), (GF1, GF2, GF3, GF4), (WF1, WF2, WF3, WF4), (RW1, RW2, RW3, RW4)) = plt.subplots(4, 4, figsize=(10,10))
itemize = [IB1, IB2, IB3, IB4, GF1, GF2, GF3, GF4, WF1, WF2, WF3, WF4, RW1, RW2, RW3, RW4]
type = [r'$IB_1$', r'$IB_2$', r'$IB_3$', r'$IB_4$', r'$GF_1$', r'$GF_2$', r'$GF_3$', r'$GF_4$', r'$WF_1$', r'$WF_2$', r'$WF_3$', r'$WF_4$', r'$RW_1$', r'$RW_2$', r'$RW_3$', r'$RW_4$']
n=0
fig.set_tight_layout(True)
for item in itemize:
    psd_test = PSD(Well_Behaved[n])
    length = len(psd_test)
    start = 0.001
    end = 2 * np.pi / 2
    omega = np.linspace(start, end, length)
    popt, pcov = curve_fit(S, omega, psd_test, maxfev = 200000, p0=(10,0.001))#, bounds=([0, 0], [10000, 10]))
    item.plot(omega, psd_test, color='black', alpha=0.5, label='experiment')
    item.plot(omega, S(omega, popt[0], popt[1]), color='black', linestyle='dotted', label='model fit')
    item.set_xscale('log')
    item.set_yscale('log')
    item.set_xlim(0.001,np.pi)
    item.set_ylim(0.001,3 * 10**7)
    item.set_xticks([])
    item.set_yticks([])
    item.annotate(type[n], xy=(0.8,0.9),xycoords='axes fraction', fontsize=12)
    #item.set_xlim(start, end)
    #item.set_title(item)
    item.annotate(r'$D_{\theta}=$'+ str(round(popt[0], 4)), xy=(0.05,0.11),xycoords='axes fraction', fontsize=10)
    item.annotate(r'$a_{\theta}=$'+ str(round(popt[1], 4)), xy=(0.05,0.05),xycoords='axes fraction', fontsize=10)
    if n==0:
        #item.legend(loc='upper right')
        q = 1
    if n in [0,4,8,12]:
        item.set_ylabel(r'$S_{\theta}(\omega)$')
        item.set_yticks([10**(-2),10**(-0),10**(2),10**(4), 10**(6)])
        #item.set_yticks([10**(-1), 10**(1), 10**(3)])
    if n in [12,13,14,15]:
        item.set_xlabel(r'$\omega$')
        item.set_xticks([10**(-2), 10**(-1), 10**(0)])
    n += 1
    #plt.savefig("FIG_D_r/" + str(testbee))
    #plt.close()
plt.show()

"""
fig, ((IB1, IB2, IB3, IB4), (GF1, GF2, GF3, GF4), (WF1, WF2, WF3, WF4), (RW1, RW2, RW3, RW4)) = plt.subplots(4, 4, figsize=(10,10))
itemize = [IB1, IB2, IB3, IB4, GF1, GF2, GF3, GF4, WF1, WF2, WF3, WF4, RW1, RW2, RW3, RW4]
n=0
fig.set_tight_layout(True)
for item in itemize:
    listy = MSD_temporal(Well_Behaved[n])
    listx = np.linspace(1, len(listy), len(listy))
    listyCUT = listy[0:20]
    listxCUT = np.linspace(1, len(listyCUT), len(listyCUT))
    popt, pcov = curve_fit(myExpFunc, listxCUT, listyCUT)
    item.plot(listx, myExpFunc(listx, *popt), label='fit', color='black', linestyle='dotted')
    item.plot(listx, listy, label='MSD', color='black', alpha=0.7)
    #item.set_xscale('log')
    #item.set_yscale('log')
    item.set_ylim(0.001, 4000)
    item.set_xticks([])
    item.set_yticks([])


    if n==0:
        item.legend(loc='center right')
    if n==0 or n==4 or n==8 or n==12:
        item.set_ylabel(r'$\langle r^{2}(\tau) \rangle$')
        item.set_yticks([0.1,3000])
        #item.set_yticks([1, 1000])
    if n==12 or n==13 or n==14 or n==15:
        item.set_xlabel(r'$\tau$')
        item.set_xticks([0, 600, 1200])
    n+=1
    item.text(400, 3500, r'$m= $' + str(round(popt[1], 3)))
    item.text(400, 3100,r'$b= $' + str(round(popt[0], 3)))

plt.subplots_adjust(wspace=0.1, hspace=0.1)
plt.show()
"""

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
