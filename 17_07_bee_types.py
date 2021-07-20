import pandas as pd
import numpy as np
from scipy import stats
import scipy.stats
from math import log, floor
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.mlab as mlab

head = list(pd.DataFrame(data=pd.read_csv("data_bee_types/LS_spatial_D_x.csv")))

#test_x = [60, 50, 40, 30, 20, 10, 0, 10, 20, 30, 40, 50, 60, 50, 40, 30, 20, 10, 0, 10, 20, 30, 40, 50, 60, 50, 40, 30, 20, 10, 0, 10, 20, 30, 40, 50, 60]
#test_y = [30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30]
testbee = "BT01A-1"



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

#theta in respect of the gradient
def theta_in_G(item):
    list_theta_in_G = []
    theta = calc_theta(item)
    dir_G = calc_dir_G(item)
    n = len(dir_G)
    for t in range(n):
        list_theta_in_G.append(theta[t] - dir_G[t])
    return list_theta_in_G
#plt.plot(theta_in_G(testbee))

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

#theta_FFT = np.fft.fft(theta_in_G(testbee))
theta_FFT = np.fft.fft(recalc_w_2pi(testbee), norm="ortho") #either norm="ortho" here or divided by C_1 in PSD()
#plt.plot(theta_FFT)

def PSD(item):
    PSD_list = []
    for i in range(round(len(theta_FFT)/2)):
        C_1 = 1#/len(theta_FFT)
        PSD_list.append(C_1 * (theta_FFT[i] * np.conj(theta_FFT[i])))
    return PSD_list
#
length = len(PSD(testbee))
#len = len(PSD(testbee))

omega = np.linspace(0, 100, length)
plt.xscale('log')
plt.yscale('log')
plt.plot(omega, PSD(testbee))


# x, y = mlab.psd(theta_FFT)[0], mlab.psd(theta_FFT)[1]
# plt.xscale('log')
# plt.yscale('log')
# plt.plot(y, x)

"""TODO: function for S_theta (loop and list) and least squares (compare PSD() and S_theta, vary D and a)!!!"""

D_r = 1000
a = 1
S_theta = 2 * D_r / (a ** 2 + omega ** 2)
plt.plot(omega, S_theta)



plt.title(testbee)
plt.show()












# print(calc_theta("BT01A-1"))
# plt.plot(calc_theta("BT01A-1"))
# plt.show()
# #calc_theta("BT14B-2")
# #calc_theta("BT01A-1")
# #plt.plot(calc_theta("BT01A-1"))
# theta_FFT = np.fft.fft(calc_theta("BT01A-1"))
# plt.plot(theta_FFT)
# plt.xlabel("")
# plt.ylabel("A")
# plt.show()
#
#
# PSD = []
# for i in range(len(theta_FFT)):
#     PSD.append((1/len(theta_FFT)) * (theta_FFT[i].real ** 2 + theta_FFT[i].imag ** 2))
# plt.plot(PSD)
# #print(len(theta_FFT))
# #plt.psd(theta_FFT)
# plt.show()
#
#
# x, y = mlab.psd(theta_FFT)[0], mlab.psd(theta_FFT)[1]
# plt.plot(y, x)
# plt.show()
# timedomain = N * dt
# omega = 2 * np.pi() * k
#
# theta_Spectral = 1/timedomain * (Re(theta_FFT()) ** 2 + Im(theta_FFT()) ** 2)#betragsquadrat von theta FFT
