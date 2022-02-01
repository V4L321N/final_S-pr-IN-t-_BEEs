import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.patches as patches
from scipy.optimize import curve_fit
from scipy.stats import norm


ArenaX = pd.DataFrame(data=pd.read_csv("~/Desktop/BAC_bee_mobility/final_S(pr)IN(t)_BEEs/V_3/data_bee_types/Arena.csv"), columns=["ArenaX"])
ArenaY = pd.DataFrame(data=pd.read_csv("~/Desktop/BAC_bee_mobility/final_S(pr)IN(t)_BEEs/V_3/data_bee_types/Arena.csv"), columns=["ArenaY"])
GradientX = pd.DataFrame(data=pd.read_csv("~/Desktop/BAC_bee_mobility/final_S(pr)IN(t)_BEEs/V_3/data_bee_types/Arena.csv"), columns=["GradX"])
GradientY = pd.DataFrame(data=pd.read_csv("~/Desktop/BAC_bee_mobility/final_S(pr)IN(t)_BEEs/V_3/data_bee_types/Arena.csv"), columns=["GradY"])

head = list(pd.DataFrame(data=pd.read_csv("~/Desktop/BAC_bee_mobility/final_S(pr)IN(t)_BEEs/V_3/data_bee_types/LS_spatial_D_x.csv")))

control_30_30 = ["BT17A-1", "BT17A-2", "BT17A-3", "BT17A-4", "BT17B-1", "BT17B-2", "BT17B-3", "BT17B-4"]
control_36_36 = ["BT07A-1", "BT07A-2", "BT07A-3", "BT07A-4", "BT07B-1", "BT07B-2", "BT07B-3", "BT07B-4"]

control_ALL = control_30_30 + control_36_36
for item in control_ALL:
    head.remove(item)

NARROW = ["BT01A-1","BT01A-2","BT01A-3","BT01A-4","BT01C-1","BT01C-2","BT01C-3","BT01C-4","BT02A-1","BT02A-2","BT02A-3","BT02A-4","BT02B-1","BT02B-2","BT02B-3","BT02B-4","BT03A-1","BT03A-2","BT03A-3","BT03A-4","BT03B-1","BT03B-2","BT03B-3","BT03B-4","BT06A-1","BT06A-2","BT06A-3","BT06A-4","BT06B-1","BT06B-2","BT06B-3","BT06B-4","BT18A-1","BT18A-2","BT18A-3","BT18A-4","BT18B-1","BT18B-2","BT18B-3","BT18B-4"]
STEEP = ["BT04A-1","BT04A-2","BT04A-3","BT04A-4","BT04B-1","BT04B-2","BT04B-3","BT04B-4","BT05A-1","BT05A-2","BT05A-3","BT05A-4","BT05B-1","BT05B-2","BT05B-3","BT05B-4","BT08A-1","BT08A-2","BT08A-3","BT08A-4","BT08B-1","BT08B-2","BT08B-3","BT08B-4","BT09A-1","BT09A-2","BT09A-3","BT09A-4","BT09B-1","BT09B-2","BT09B-3","BT09B-4","BT10A-1","BT10A-2","BT10A-3","BT10A-4","BT10B-1","BT10B-2","BT10B-3","BT10B-4"]
STEEPEST = ["BT11A-1","BT11A-2","BT11A-3","BT11A-4","BT11B-1","BT11B-2","BT11B-3","BT11B-4","BT12A-1","BT12A-2","BT12A-3","BT12A-4","BT12B-1","BT12B-2","BT12B-3","BT12B-4","BT13A-1","BT13A-2","BT13A-3","BT13A-4","BT13B-1","BT13B-2","BT13B-3","BT13B-4","BT14A-1","BT14A-2","BT14A-3","BT14A-4","BT14B-1","BT14B-2","BT14B-3","BT14B-4","BT15A-1","BT15A-2","BT15A-3","BT15A-4","BT15B-1","BT15B-2","BT15B-3","BT15B-4"]

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

"""-begin------function: remove empty entires XY and T-----------------------"""
def X_remove_NaN(item):
    X_dataset_entropy = pd.DataFrame(data=pd.read_csv("~/Desktop/BAC_bee_mobility/final_S(pr)IN(t)_BEEs/V_3/data_bee_types/LS_spatial_D_x.csv"), columns=[item])
    X_dataset_wo_NAN = []
    for m in range(len(X_dataset_entropy)):
        check = X_dataset_entropy.iloc[m].item()
        if str(check) != "nan":
            X_dataset_wo_NAN.append(check)
    return X_dataset_wo_NAN

def Y_remove_NaN(item):
    Y_dataset_entropy = pd.DataFrame(data=pd.read_csv("~/Desktop/BAC_bee_mobility/final_S(pr)IN(t)_BEEs/V_3/data_bee_types/LS_spatial_D_y.csv"), columns=[item])
    Y_dataset_wo_NAN = []
    for m in range(len(Y_dataset_entropy)):
        check = Y_dataset_entropy.iloc[m].item()
        if str(check) != "nan":
            Y_dataset_wo_NAN.append(check)
    return Y_dataset_wo_NAN

def T_remove_NaN(item):
    T_dataset = pd.DataFrame(data=pd.read_csv("~/Desktop/BAC_bee_mobility/final_S(pr)IN(t)_BEEs/V_3/data_bee_types/temperature_runs.csv"), columns=[item])
    T_dataset_wo_NAN = []
    for m in range(1, len(T_dataset)):
        check = T_dataset.iloc[m].item()
        if str(check) != "nan":
            T_dataset_wo_NAN.append(check)
    return T_dataset_wo_NAN
"""-end--------function: remove empty entires XY and T-----------------------"""

# """-begin------figure: TRAJECTORIES OF 16 EXEMPLARY BEES---------------------"""
# fig, ((IB1, IB2, IB3, IB4), (GF1, GF2, GF3, GF4), (WF1, WF2, WF3, WF4), (RW1, RW2, RW3, RW4)) = plt.subplots(4, 4, figsize=(10,10))
# itemize = [IB1, IB2, IB3, IB4, GF1, GF2, GF3, GF4, WF1, WF2, WF3, WF4, RW1, RW2, RW3, RW4]
# type = [r'$IB_1$', r'$IB_2$', r'$IB_3$', r'$IB_4$', r'$GF_1$', r'$GF_2$', r'$GF_3$', r'$GF_4$', r'$WF_1$', r'$WF_2$', r'$WF_3$', r'$WF_4$', r'$RW_1$', r'$RW_2$', r'$RW_3$', r'$RW_4$']
# n=0
# fig.set_tight_layout(True)
# for item in itemize:
#     x,y = X_remove_NaN(Well_Behaved[n]),Y_remove_NaN(Well_Behaved[n])
#     item.plot(ArenaX, ArenaY, color="black", linestyle="dashed")
#     item.plot(GradientX, GradientY, color="black", linestyle="dotted", dash_capstyle='round')
#     item.annotate(type[n], xy=(0.8,0.9),xycoords='axes fraction', fontsize=12)
#     item.set_xticks([])
#     item.set_yticks([])
#     R = 0.0
#     G = 0.0
#     B = 1.0
#     time = len(x)
#     for i in range(time-1):
#         item.plot((x[i],x[i+1]),(y[i],y[i+1]),color=(R,G,B))
#         R+=(1/time)
#         #G+=0.001
#         B-=(1/time)
#     n += 1
# plt.show()
# """-end------figure: TRAJECTORIES OF 16 EXEMPLARY BEES-----------------------"""

"""-begin----------function: different fitting functions---------------------"""
def myExpFunc(x, k1, k2):
    return x ** k1 + k2

# def myExpFunc(x, k1, k2):     # only for comparison with function above
#     return x * np.exp(k1) + k2

def myLinFunc(x, b, m):
    return x * m + b

def myLin2Func(x, m):
    return m * x

def myPower(x,c1,c2,c3):
    return c3 * x ** (c1) + c2

def myPoly(x,q1,q2,q3):
    return x * q1 + x**2 * q2**2 + q3

# def myCurve(x, tau, beta):    # only for comparison with function below
#     return x ** tau * beta

def myCurve(x,tau,beta):
    return tau * np.exp(-beta * x)

def myCurve1(x,tau,beta):
    return  tau / (1 + (beta * x))

def myCurve2(x,tau,beta):
    return  tau * np.exp(-beta * x)

def myCurve3(x,tau,beta,gamma):
    return tau * (beta * x) ** gamma

def powerCurve(f, alpha):
    return 1/(f**alpha)
"""-end------------function: different fitting functions---------------------"""

# """--begin-------figure: MSD BRANCHES 16 BEES--------------------------------"""
# def MSD_bundle(bee, item):
#     lin = np.linspace(0,500,10)
#     x,y = X_remove_NaN(bee),Y_remove_NaN(bee)
#     posx = 0
#     posy = 0
#     MSD = [0]
#     slope = []
#     for i in range(1,len(x)):
#         if np.sqrt((x[i]-30)**2+(y[i]-30)**2) > 29:
#             posx = 0
#             posy = 0
#             item.plot(MSD, color='black', alpha=0.3)
#             slope.append(MSD[-1]/len(MSD))
#             MSD = [0]
#         else:
#             MSD.append(np.sqrt(posx**2 + posy**2))
#             posx += np.sqrt((x[i-1]-x[i])**2)
#             posy += np.sqrt((y[i-1]-y[i])**2)
#     slope.append(MSD[-1]/len(MSD))
#     item.plot(MSD, color='black', alpha=0.3)
#     item.plot(lin,lin,linestyle='dotted',color='black')
#     item.set_xticks([])
#     item.set_yticks([])
#     item.set_xlim(-10,510)
#     item.set_ylim(-10,510)
#     item.annotate(type[n], xy=(0.8,0.9),xycoords='axes fraction', fontsize=12)
#     if n in [0,4,8,12]:
#         item.set_ylabel(r'$\langle r^{2}(\tau) \rangle$')
#         item.set_yticks([250,500])
#     if n in [12,13,14,15]:
#         item.set_xlabel(r'$\tau$')
#         item.set_xticks([0, 250, 500])
#
# fig, ((IB1, IB2, IB3, IB4), (GF1, GF2, GF3, GF4), (WF1, WF2, WF3, WF4), (RW1, RW2, RW3, RW4)) = plt.subplots(4, 4, figsize=(10,10))
# itemize = [IB1, IB2, IB3, IB4, GF1, GF2, GF3, GF4, WF1, WF2, WF3, WF4, RW1, RW2, RW3, RW4]
# type = [r'$IB_1$', r'$IB_2$', r'$IB_3$', r'$IB_4$', r'$GF_1$', r'$GF_2$', r'$GF_3$', r'$GF_4$', r'$WF_1$', r'$WF_2$', r'$WF_3$', r'$WF_4$', r'$RW_1$', r'$RW_2$', r'$RW_3$', r'$RW_4$']
# n=0
# fig.set_tight_layout(True)
# for i in itemize:
#     MSD_bundle(Well_Behaved[n], i)
#     n += 1
# plt.show()
# """-end----------figure: MSD BRANCHES 16 BEES--------------------------------"""
#
"""-begin--------function: calculate TSMD------------------------------------"""
def MSD_temporal(item):
    x,y = X_remove_NaN(item),Y_remove_NaN(item)
    MSD = []
    length = len(x)
    for tau in range(1, length):
        MSD_tau = 0
        for t in range(0, (length-tau)):
            MSD_tau += (x[t+tau]-x[t])**2 + (y[t+tau]-y[t])**2
        MSD.append(MSD_tau/(length-tau))
    return MSD
"""-end-----------function: calculate TSMD-----------------------------------"""
#
# """-begin--------figure: TSMD 16 BEES/LINEAR PLOT----------------------------"""
# fig, ((IB1, IB2, IB3, IB4), (GF1, GF2, GF3, GF4), (WF1, WF2, WF3, WF4), (RW1, RW2, RW3, RW4)) = plt.subplots(4, 4, figsize=(10,10))
# itemize = [IB1, IB2, IB3, IB4, GF1, GF2, GF3, GF4, WF1, WF2, WF3, WF4, RW1, RW2, RW3, RW4]
# n=0
# fig.set_tight_layout(True)
# for item in itemize:
#     listy = MSD_temporal(Well_Behaved[n])
#     listx = np.linspace(1, len(listy), len(listy))
#     listyCUT = listy[0:20]
#     listxCUT = np.linspace(1, len(listyCUT), len(listyCUT))
#     popt, pcov = curve_fit(myLinFunc, listxCUT, listyCUT)
#     item.plot(listx, myLinFunc(listx, *popt), label='fit', color='black', linestyle='dotted')
#     item.plot(listx, listy, label='MSD', color='black', alpha=0.7)
#     #item.set_xscale('log')
#     #item.set_yscale('log')
#     item.set_ylim(0.001, 4000)
#     item.set_xticks([])
#     item.set_yticks([])
#     if n==0:
#         item.legend(loc='center right')
#     if n==0 or n==4 or n==8 or n==12:
#         item.set_ylabel(r'$\langle r^{2}(\tau) \rangle$')
#         item.set_yticks([0.1,3000])
#         #item.set_yticks([1, 1000])
#     if n==12 or n==13 or n==14 or n==15:
#         item.set_xlabel(r'$\tau$')
#         item.set_xticks([0, 600, 1200])
#     n+=1
#     item.text(400, 3500, r'$m= $' + str(round(popt[1], 3)))
#     item.text(400, 3100,r'$b= $' + str(round(popt[0], 3)))
# plt.subplots_adjust(wspace=0.1, hspace=0.1)
# plt.show()
# """-end----------figure: TSMD 16 BEES/LINEAR PLOT----------------------------"""
#
# """-begin----------figure: TSMD DIFFERENT FITS 16 BEES/LOGLOG PLOT-----------"""
# fig, ((IB1, IB2, IB3, IB4), (GF1, GF2, GF3, GF4), (WF1, WF2, WF3, WF4), (RW1, RW2, RW3, RW4)) = plt.subplots(4, 4, figsize=(10,10))
# itemize = [IB1, IB2, IB3, IB4, GF1, GF2, GF3, GF4, WF1, WF2, WF3, WF4, RW1, RW2, RW3, RW4]
# type = [r'$IB_1$', r'$IB_2$', r'$IB_3$', r'$IB_4$', r'$GF_1$', r'$GF_2$', r'$GF_3$', r'$GF_4$', r'$WF_1$', r'$WF_2$', r'$WF_3$', r'$WF_4$', r'$RW_1$', r'$RW_2$', r'$RW_3$', r'$RW_4$']
# n=0
# fig.set_tight_layout(True)
# for item in itemize:#head:
#     listy = MSD_temporal(Well_Behaved[n])
#     listx = np.linspace(0, len(listy), len(listy))
#     item.annotate(type[n], xy=(0.8,0.9),xycoords='axes fraction', fontsize=12)
#     item.plot(listx, listy, label='MSD', color='black', alpha=0.5)
#
#     START_poly = 0#20
#     END_poly = 10#len(listy)
#     listyPOLY = listy[START_poly:END_poly]
#     listxPOLY = np.linspace(START_poly, END_poly, END_poly-START_poly)
#     popt_poly, pcov_poly = curve_fit(myPoly, listxPOLY, listyPOLY, maxfev = 2000000, p0=(1, 350, 1))
#     item.plot(listx, myPoly(listx, *popt_poly), label='polynomial fit', color='black', linestyle='dashed')
#
#     START_power = 0#20
#     END_power = 10#len(listy)
#     listyPOWER = listy[START_power:END_power]
#     listxPOWER = np.linspace(START_power, END_power, END_power-START_power)
#     popt_power, pcov_power = curve_fit(myPower, listxPOWER, listyPOWER, maxfev = 2000000, p0=(1, 350, 1))
#     item.plot(listx, myPower(listx, *popt_power), label='power law fit', color='red', linestyle='dashdot')
#
#     START_lin = 0#20
#     END_lin = 20#len(listy)
#     listyLIN = listy[START_lin:END_lin]
#     listxLIN = np.linspace(START_lin, END_lin, END_lin-START_lin)
#     popt_lin, pcov_lin = curve_fit(myLin2Func, listxLIN, listyLIN, maxfev = 200000, p0=(1))
#     item.plot(listx, myLin2Func(listx, *popt_lin), label='linear fit', color='blue', linestyle='dotted')#, linewidth=1)
#
#     item.plot(listx, (listx/listx) * listy[1], color='black', alpha=0.5, linewidth=0.5, linestyle='dotted')
#     item.text(30, listy[1] + listy[1]/10, r'$I = $' + str(round(listy[1], 3)))
#     item.set_xscale('log')
#     item.set_yscale('log')
#     item.set_ylim(0.1,4000)
#     item.set_xticks([])
#     item.set_yticks([])
#
#     if n in [0,1,2,3]:
#         item.set_ylim(0.001,4000)
#         item.annotate(r'$v = $'+ str(round(popt_poly[1],3)), xy=(0.1,0.7), xycoords='axes fraction', fontsize=10)
#         item.annotate(r'$\alpha = $'+ str(round(popt_power[0],3)), xy=(0.1,0.9), xycoords='axes fraction', fontsize=10)
#         item.annotate(r'$K_{\alpha} = $'+ str(round(popt_power[2],3)), xy=(0.1,0.8), xycoords='axes fraction', fontsize=10)
#     if n in [4,5,6,7]:
#         item.annotate(r'$v = $'+ str(round(popt_poly[1],3)), xy=(0.1,0.7), xycoords='axes fraction', fontsize=10)
#         item.annotate(r'$\alpha = $'+ str(round(popt_power[0],3)), xy=(0.1,0.9), xycoords='axes fraction', fontsize=10)
#         item.annotate(r'$K_{\alpha} = $'+ str(round(popt_power[2],3)), xy=(0.1,0.8), xycoords='axes fraction', fontsize=10)
#     if n in [8,9,10,11,12,13,14,15]:
#         item.annotate(r'$v = $'+ str(round(popt_poly[1],3)), xy=(0.1,0.05), xycoords='axes fraction', fontsize=10)
#         item.annotate(r'$\alpha = $'+ str(round(popt_power[0],3)), xy=(0.1,0.25), xycoords='axes fraction', fontsize=10)
#         item.annotate(r'$K_{\alpha} = $'+ str(round(popt_power[2],3)), xy=(0.1,0.15), xycoords='axes fraction', fontsize=10)
#     if n == 0:
#         item.set_ylabel(r'$\langle r^{2}(\tau) \rangle$')
#         item.set_yticks([10**(-3),10**(-1),10**(1),10**(3)])
#     if n in [4,8,12]:
#         item.set_ylabel(r'$\langle r^{2}(\tau) \rangle$')
#         item.set_yticks([10**(-1),10**(1),10**(3)])
#         #item.set_yticks([1, 1000])
#     if n in [12,13,14,15]:
#         item.set_xlabel(r'$\tau$')
#         item.set_xticks([10**(-3),10**(-1),10**(1),10**(3)])
#     n+=1
#     item.set_xlim(1,1200)
# plt.show()
# """-end------------figure: TSMD DIFFERENT FITS 16 BEES/LOGLOG PLOT-----------"""
#
# """-begin----------figure: TSMD TWO FITS ALL BEES/LOGLOG PLOT----------------"""
# inter_list = []
# exp_list = []
# for item in head:
#     listy = MSD_temporal(item)
#     listx = np.linspace(0, len(listy), len(listy))
#     plt.plot(listx, listy, color='black', alpha=0.5, linewidth=0.3)
#     inter_list.append(listy[1])
#     START_lin = 0
#     END_lin = 20 #len(listy)
#     listyLIN = listy[START_lin:END_lin]
#     listxLIN = np.linspace(START_lin, END_lin, END_lin-START_lin)
#     popt_lin, pcov_lin = curve_fit(myExpFunc, listxLIN, listyLIN, maxfev = 200000, p0=(2, 30))
#     exp_list.append(popt_lin[0])
#     plt.xscale('log')
#     plt.yscale('log')
#     plt.xlim(1,1200)
# std_inter = np.std(inter_list)
# mean_inter = np.mean(inter_list)
# mean_exp = np.mean(exp_list)
# plt.plot(listx, (listx/listx) * mean_inter, color='red', alpha=0.8, linewidth=1.3, linestyle='dotted')
# plt.plot(listx, listx**mean_exp, color='blue', alpha=0.8, linewidth=1.1)
# plt.annotate(r'$m = $'+ str(round(mean_inter,3)), xy=(0.05,0.85), xycoords='axes fraction', fontsize=10)
# plt.annotate(r'$\alpha = $'+ str(round(mean_exp,3)), xy=(0.05,0.9), xycoords='axes fraction', fontsize=10)
# plt.xlabel(r'$\tau$')
# plt.ylabel(r'$\langle r^{2}(\tau) \rangle$')
# plt.show()
# """-end----------figure: TSMD TWO FITS ALL BEES/LOGLOG PLOT----------------"""

"""-begin----------function: calculate sitting and walking velocites------"""
def calc_vel_sw(item):
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
"""-end----------function: calculate mean sitting and walking velocites------"""

# """-begin--------figure: HISTOGRAM OF THE VELOCITIES 16 BEES-----------------"""
# fig, ((IB1, IB2, IB3, IB4), (GF1, GF2, GF3, GF4), (WF1, WF2, WF3, WF4), (RW1, RW2, RW3, RW4)) = plt.subplots(4, 4, figsize=(10,10))
# itemize = [IB1, IB2, IB3, IB4, GF1, GF2, GF3, GF4, WF1, WF2, WF3, WF4, RW1, RW2, RW3, RW4]
# type = [r'$IB_1$', r'$IB_2$', r'$IB_3$', r'$IB_4$', r'$GF_1$', r'$GF_2$', r'$GF_3$', r'$GF_4$', r'$WF_1$', r'$WF_2$', r'$WF_3$', r'$WF_4$', r'$RW_1$', r'$RW_2$', r'$RW_3$', r'$RW_4$']
# n=0
# fig.set_tight_layout(True)
# for item in itemize:
#     item.set_xlim(0, 6.1, 0.1)
#     item.set_ylim(0,5.1)
#     item.hist(calc_vel_sw(Well_Behaved[n])[0], bins=30, range=(0,6), density=True, alpha=0.50, color='blue', label='valid data')
#     item.hist(calc_vel_sw(Well_Behaved[n])[1], bins=30, range=(0,6), density=True, alpha=0.50, color='red', label='removed data')
#     std_walk = np.std(calc_vel_sw(Well_Behaved[n])[0], ddof=1)
#     mean_walk = np.mean(calc_vel_sw(Well_Behaved[n])[0])
#     std_stop = np.std(calc_vel_sw(Well_Behaved[n])[1], ddof=1)
#     mean_stop = np.mean(calc_vel_sw(testbee)[1]) #0
#     domain = np.linspace(0, 6)
#     item.set_xticks([])
#     item.set_yticks([])
#     item.annotate(type[n], xy=(0.8,0.9),xycoords='axes fraction', fontsize=12)
#     item.plot(domain, norm.pdf(domain, mean_walk, std_walk), color='black', linestyle='dashed')
#     item.plot(domain, norm.pdf(domain, mean_stop, std_stop), color='red', linestyle='dashed')
#     item.annotate(r'$\mu \approx $'+ str(round(mean_walk, 3)) + ' ' + r'$cm/s$', xy=(0.1,0.9), xycoords='axes fraction', fontsize=10)
#     item.annotate(r'$\sigma \approx $'+ str(round(std_walk, 3)) + ' ' + r'$cm/s$', xy=(0.1,0.8), xycoords='axes fraction', fontsize=10)
#     if n in [0,4,8,12]:
#         item.set_ylabel(r'$counts$')
#         item.set_yticks([0,5])
#     if n in [12,13,14,15]:
#         item.set_xlabel('velocity ' + r'$(cm/s)$')
#         item.set_xticks([0, 2, 4, 6])
#     n += 1
# plt.show()
# """-end--------figure: HISTOGRAM OF THE VELOCITIES 16 BEES-----------------"""

"""-begin------function: calculate velocity for each time step---------------"""
def calc_vel(item):
    list_vel = []
    X_dataset_wo_NAN = X_remove_NaN(item) #test_x
    Y_dataset_wo_NAN = Y_remove_NaN(item) #test_y
    n = len(X_dataset_wo_NAN) - 1
    for t in range(n):
        velocity = np.sqrt((X_dataset_wo_NAN[t + 1] - X_dataset_wo_NAN[t]) ** 2 + (Y_dataset_wo_NAN[t + 1] - Y_dataset_wo_NAN[t]) ** 2)
        list_vel.append(velocity)
    return list_vel
"""-end--------function: calculate velocity in for each time step------------"""

"""-begin--------function: calculate fourier transform of the velocity-------"""
def velocity_FFT(item):
    return np.fft.fft(calc_vel(item), norm="ortho") #either norm="ortho" here or divided by C_2 in PSD()
"""-end----------function: calculate fourier transform of the velocity-------"""

"""-begin--------function: calculate power spectral density of the FT--------"""
def PSD_vel(item):
    vel_FFT = velocity_FFT(item)
    loop_L = round(len(vel_FFT)/2)
    PSD_list = []
    for i in range(loop_L):
        C_2 = 1
        absolute = C_2 * (vel_FFT[i] * np.conj(vel_FFT[i]))
        PSD_list.append(absolute.real)
    return PSD_list
"""-end----------function: calculate power spectral density of the angle FT--"""

"""-begin--------function: analytical solution for the PSD-------------------"""
def S(w, D, coupling_a):
    return 2 * D / (coupling_a ** 2 + w ** 2)
"""-end----------function: analytical solution for the PSD-------------------"""

# """-begin--------figure: PSD OF THE VELOCITY 16 BEES-------------------------"""
# fig, ((IB1, IB2, IB3, IB4), (GF1, GF2, GF3, GF4), (WF1, WF2, WF3, WF4), (RW1, RW2, RW3, RW4)) = plt.subplots(4, 4, figsize=(10,10))
# itemize = [IB1, IB2, IB3, IB4, GF1, GF2, GF3, GF4, WF1, WF2, WF3, WF4, RW1, RW2, RW3, RW4]
# type = [r'$IB_1$', r'$IB_2$', r'$IB_3$', r'$IB_4$', r'$GF_1$', r'$GF_2$', r'$GF_3$', r'$GF_4$', r'$WF_1$', r'$WF_2$', r'$WF_3$', r'$WF_4$', r'$RW_1$', r'$RW_2$', r'$RW_3$', r'$RW_4$']
# n=0
# fig.set_tight_layout(True)
# for item in itemize:
#     psd_test = PSD_vel(Well_Behaved[n])
#     length = len(psd_test)
#     start = 0.001
#     end = 2 * np.pi / 2
#     omega = np.linspace(start, end, length)
#     popt, pcov = curve_fit(S, omega, psd_test)
#     item.plot(omega, psd_test, color='black', alpha=0.5, label='experiment')
#     item.plot(omega, S(omega, popt[0], popt[1]), color='black', linestyle='dashdot', label='model fit')
#     item.set_xscale('log')
#     item.set_yscale('log')
#     item.set_xlim(0.001,np.pi)
#     item.set_ylim(0.0001,10000)
#     item.set_xticks([])
#     item.set_yticks([])
#     item.annotate(type[n], xy=(0.8,0.9),xycoords='axes fraction', fontsize=12)
#     item.annotate(r'$D_{v}=$'+ str(round(popt[0], 4)), xy=(0.05,0.11),xycoords='axes fraction', fontsize=10)
#     item.annotate(r'$a_{v}=$'+ str(round(popt[1], 4)), xy=(0.05,0.05),xycoords='axes fraction', fontsize=10)
#     if n in [0,4,8,12]:
#         item.set_ylabel(r'$S_{v}(\omega)$')
#         item.set_yticks([10**(-2), 10**0, 10**2])
#     if n in [12,13,14,15]:
#         item.set_xlabel(r'$\omega$')
#         item.set_xticks([10**(-2),10**(-1), 10**0])
#     n += 1
# plt.show()
# """-end----------figure: PSD OF THE VELOCITY 16 BEES-------------------------"""

"""-begin------function: calculate turning angle in for each time step-------"""
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
"""-end--------function: calculate turning angle in for each time step-------"""

"""-begin------function: position of the gradient depending on bee position--"""
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
"""-end--------function: position of the gradient depending on bee position--"""

"""-begin------function: turning angle in respect of gradient position-------"""
def theta_in_G(item):
    list_theta_in_G = []
    theta = calc_theta(item)
    dir_G = calc_dir_G(item)
    n = len(dir_G)
    for t in range(n):
        list_theta_in_G.append(theta[t] - dir_G[t])
    return list_theta_in_G
"""-end--------function: turning angle in respect of gradient position-------"""

"""-begin------function: check turning angle to avoid jumps from -pi and +pi-"""
def recalc_w_2pi(item):
    theta_G = theta_in_G(item)
    n = len(theta_G)
    for t in range(n):
        while theta_G[t] - theta_G[t-1] > np.pi:
            theta_G[t] -= 2 * np.pi
        while theta_G[t] - theta_G[t-1] < -np.pi:
            theta_G[t] += 2 * np.pi
    return theta_G
"""-end--------function: check turning angle to avoid jumps from -pi and +pi-"""

"""-begin--------function: calculate fourier transform of the turning angle--"""
def theta_FFT(item):
    return np.fft.fft(recalc_w_2pi(item), norm="ortho")
"""-end----------function: calculate fourier transform of the turning angle--"""

"""-begin--------function: calculate power spectral density of the FT--------"""
def PSD_theta(item):
    th_FFT = theta_FFT(item)
    loop_L = round(len(th_FFT)/2)
    PSD_list = []
    for i in range(loop_L):
        C_1 = 1#/len(theta_FFT)
        absolute = C_1 * (th_FFT[i] * np.conj(th_FFT[i]))
        PSD_list.append(absolute.real)
    return PSD_list
"""-end----------function: calculate power spectral density of the FT--------"""

# """-begin--------figure: PSD OF ANGLE THETA 16 BEES--------------------------"""
# fig, ((IB1, IB2, IB3, IB4), (GF1, GF2, GF3, GF4), (WF1, WF2, WF3, WF4), (RW1, RW2, RW3, RW4)) = plt.subplots(4, 4, figsize=(10,10))
# itemize = [IB1, IB2, IB3, IB4, GF1, GF2, GF3, GF4, WF1, WF2, WF3, WF4, RW1, RW2, RW3, RW4]
# type = [r'$IB_1$', r'$IB_2$', r'$IB_3$', r'$IB_4$', r'$GF_1$', r'$GF_2$', r'$GF_3$', r'$GF_4$', r'$WF_1$', r'$WF_2$', r'$WF_3$', r'$WF_4$', r'$RW_1$', r'$RW_2$', r'$RW_3$', r'$RW_4$']
# n=0
# fig.set_tight_layout(True)
# for item in itemize:
#     psd_test = PSD_theta(Well_Behaved[n])
#     length = len(psd_test)
#     start = 0.001
#     end = 2 * np.pi / 2
#     omega = np.linspace(start, end, length)
#     popt, pcov = curve_fit(S, omega, psd_test, maxfev = 200000, p0=(10,0.001))#, bounds=([0, 0], [10000, 10]))
#     item.plot(omega, psd_test, color='black', alpha=0.5, label='experiment')
#     item.plot(omega, S(omega, popt[0], popt[1]), color='black', linestyle='dotted', label='model fit')
#     item.set_xscale('log')
#     item.set_yscale('log')
#     item.set_xlim(0.001,np.pi)
#     item.set_ylim(0.001,3 * 10**7)
#     item.set_xticks([])
#     item.set_yticks([])
#     item.annotate(type[n], xy=(0.8,0.9),xycoords='axes fraction', fontsize=12)
#     item.annotate(r'$D_{\theta}=$'+ str(round(popt[0], 4)), xy=(0.05,0.11),xycoords='axes fraction', fontsize=10)
#     item.annotate(r'$a_{\theta}=$'+ str(round(popt[1], 4)), xy=(0.05,0.05),xycoords='axes fraction', fontsize=10)
#     if n==0:
#         q = 1
#     if n in [0,4,8,12]:
#         item.set_ylabel(r'$S_{\theta}(\omega)$')
#         item.set_yticks([10**(-2),10**(-0),10**(2),10**(4), 10**(6)])
#     if n in [12,13,14,15]:
#         item.set_xlabel(r'$\omega$')
#         item.set_xticks([10**(-2), 10**(-1), 10**(0)])
#     n += 1
# plt.show()
# """-end----------figure: PSD OF ANGLE THETA 16 BEES--------------------------"""

# """-begin--------figure: A THETA OVER D THETA ALL BEES/LOGLIN----------------"""
# ALL_a_list = []
# ALL_D_r_list = []
# for testbee in head:
#     psd_test = PSD_theta(testbee)
#     length = len(psd_test)
#     start = 0.001
#     end = np.pi
#     omega = np.linspace(start, end, length)
#     popt, pcov = curve_fit(S, omega, psd_test)
#     ALL_D_r_list.append(popt[0])
#     ALL_a_list.append(np.abs(popt[1]))
# plt.scatter(ALL_D_r_list, ALL_a_list, color='black', alpha=0.5)
# plt.xscale('log')
# plt.ylabel(r'$a_{\theta}$')
# plt.xlabel(r'$D_{\theta}$')
# n=0
# for testbee in head:
#     if testbee in Well_Behaved:
#         if testbee in ["BT06A-2", "BT06A-3", "BT02B-2", "BT02B-1"]:
#             plt.scatter(ALL_D_r_list[n], ALL_a_list[n], color='blue', alpha=1, marker='X', s=70)#, label=r'$IB_{1-4}$')
#         if testbee in ["BT09A-2", "BT09B-2", "BT09B-4", "BT12B-2"]:
#             plt.scatter(ALL_D_r_list[n], ALL_a_list[n], color='red', alpha=1, marker='X', s=70)#, label=r'$GF_{1-4}$')
#         if testbee in ["BT04B-3", "BT09B-1", "BT12B-1", "BT13B-3"]:
#             plt.scatter(ALL_D_r_list[n], ALL_a_list[n], color='blue', alpha=1, marker='P', s=70)#, label=r'$WF_{1-4}$')
#         if testbee in ["BT03A-1", "BT06B-1", "BT13A-3", "BT12A-1"]:
#             plt.scatter(ALL_D_r_list[n], ALL_a_list[n], color='red', alpha=1, marker='P', s=70)#, label=r'$RW_{1-4}$')
#     n+=1
# plt.show()
# """-end----------figure: A THETA OVER D THETA ALL BEES/LOGLIN----------------"""

"""-begin------function: map velocity as shot noise--------------------------"""
def calc_vel_shot(item):
    list_vel = []
    list_shot = []
    X_dataset_wo_NAN = X_remove_NaN(item)
    Y_dataset_wo_NAN = Y_remove_NaN(item)
    n = len(X_dataset_wo_NAN) - 1
    for t in range(n):
        velocity = np.sqrt((X_dataset_wo_NAN[t + 1] - X_dataset_wo_NAN[t]) ** 2 + (Y_dataset_wo_NAN[t + 1] - Y_dataset_wo_NAN[t]) ** 2)
        if velocity <= 0.5:
            list_shot.append(0)
        else:
            list_shot.append(1)
        list_vel.append(velocity)
    return list_vel, list_shot
"""-end--------function: map velocity as shot noise--------------------------"""

# """-begin------figures: RTS VELOCITIES---------------------------------------"""
# first_squad = [IB_1, IB_2, IB_3, IB_4, GF_1, GF_2, GF_3, GF_4]
# second_squad = [WF_1, WF_2, WF_3, WF_4, RW_1, RW_2, RW_3, RW_4]
# fig, ((IB1, IB2, IB3, IB4, GF1, GF2, GF3, GF4)) = plt.subplots(8, 1, figsize=(10,10))
# itemize = [IB1, IB2, IB3, IB4, GF1, GF2, GF3, GF4]
# type = [r'$IB_1$', r'$IB_2$', r'$IB_3$', r'$IB_4$', r'$GF_1$', r'$GF_2$', r'$GF_3$', r'$GF_4$']
# m=0
# fig.set_tight_layout(True)
# for item in itemize:
#     vel, shot = calc_vel_shot(first_squad[m])
#     color_RTS = 'tab:red'
#     item.set_ylabel('RTS', color=color_RTS, fontdict=dict(weight='bold'))
#     item.plot(shot, color=color_RTS)
#     ax2 = item.twinx()
#     color_vel = 'tab:blue'
#     ax2.set_ylabel('v'+r'$ [cm/s]$', color=color_vel, fontdict=dict(weight='bold'))
#     ax2.plot(vel, color=color_vel)
#     ax2.set_ylim(-0.2,6.2)
#     ax2.set_yticks([0,3,6])
#     fig.tight_layout()
#     item.set_xlim(-1,1201)
#     item.set_ylim(-0.04, 1.04)
#     item.set_xticks([])
#     item.set_yticks([0,1])
#     if item == GF4:
#         item.set_xticks([0,1200])
#         item.set_xlabel('time '+r'$ [s]$')
#     item.annotate(type[m], xy=(0.96,0.8),xycoords='axes fraction', fontsize=10)
#     m+=1
# plt.show()
# fig, ((WF1, WF2, WF3, WF4, RW1, RW2, RW3, RW4)) = plt.subplots(8, 1, figsize=(10,10))
# itemize = [WF1, WF2, WF3, WF4, RW1, RW2, RW3, RW4]
# type = [r'$WF_1$', r'$WF_2$', r'$WF_3$', r'$WF_4$', r'$RW_1$', r'$RW_2$', r'$RW_3$', r'$RW_4$']
# n=0
# fig.set_tight_layout(True)
# for item in itemize:
#     vel, shot = calc_vel_shot(second_squad[n])
#     color_RTS = 'tab:red'
#     item.set_ylabel('RTS', color=color_RTS, fontdict=dict(weight='bold'))
#     item.plot(shot, color=color_RTS)
#     ax2 = item.twinx()
#     color_vel = 'tab:blue'
#     ax2.set_ylabel('v'+r'$ [cm/s]$', color=color_vel, fontdict=dict(weight='bold'))
#     ax2.plot(vel, color=color_vel)
#     ax2.set_ylim(-0.2,6.2)
#     ax2.set_yticks([0,3,6])
#     fig.tight_layout()
#     item.set_xlim(-1,1201)
#     item.set_ylim(-0.04, 1.04)
#     item.set_xticks([])
#     item.set_yticks([0,1])
#     if item == RW4:
#         item.set_xticks([0,1200])
#         item.set_xlabel('time '+r'$ [s]$')
#     item.annotate(type[n], xy=(0.96,0.8),xycoords='axes fraction', fontsize=10)
#     n+=1
# plt.show()
# """-end--------figures: RTS VELOCITIES---------------------------------------"""

"""-begin------function: calculate walking duration--------------------------"""
def calc_dur_walk(item):
    list_vel = []
    X_dataset_wo_NAN = X_remove_NaN(item)
    Y_dataset_wo_NAN = Y_remove_NaN(item)
    n = len(X_dataset_wo_NAN) - 1
    for t in range(n):
        velocity = np.sqrt((X_dataset_wo_NAN[t + 1] - X_dataset_wo_NAN[t]) ** 2 + (Y_dataset_wo_NAN[t + 1] - Y_dataset_wo_NAN[t]) ** 2)
        list_vel.append(velocity)
    T_dataset_wo_NAN = T_remove_NaN(item)
    T_minmax_data = pd.DataFrame(data=pd.read_csv("~/Desktop/BAC_bee_mobility/final_S(pr)IN(t)_BEEs/V_3/data_bee_types/temperature_variables.csv"), columns=[item])
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
"""-end--------function: calculate walking duration--------------------------"""

"""-begin------function: calculate stopping duration-------------------------"""
def calc_dur_stop(item):
    list_vel = []
    X_dataset_wo_NAN = X_remove_NaN(item)
    Y_dataset_wo_NAN = Y_remove_NaN(item)
    n = len(X_dataset_wo_NAN) - 1
    for t in range(n):
        velocity = np.sqrt((X_dataset_wo_NAN[t + 1] - X_dataset_wo_NAN[t]) ** 2 + (Y_dataset_wo_NAN[t + 1] - Y_dataset_wo_NAN[t]) ** 2)
        list_vel.append(velocity)
    T_dataset_wo_NAN = T_remove_NaN(item)
    T_minmax_data = pd.DataFrame(data=pd.read_csv("~/Desktop/BAC_bee_mobility/final_S(pr)IN(t)_BEEs/V_3/data_bee_types/temperature_variables.csv"), columns=[item])
    T_ideal = T_minmax_data.iloc[1].item()
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
                mean_delta_stop_temp.append(np.abs(np.mean(stop_temperatures) - T_ideal))
                duration_walk = 0
                walk_temperatures = []
            elif v == len(list_vel[:-2]):
                all_stop_durations.append(duration_stop)
                mean_delta_stop_temp.append(np.abs(np.mean(stop_temperatures) - T_ideal))
    return all_stop_durations, mean_delta_stop_temp
"""-end--------function: calculate stopping duration-------------------------"""

# """-begin------figure: STOPPING AND WALKING DURATIONS OVER TEMPERATURE-------"""
# ALL_WALKS_at_dT = []
# ALL_TEMPS_at_walk = []
# ALL_STOPS_at_dT = []
# ALL_TEMPS_at_stop = []
# for testbee in head:
#     walking_times, mean_walk_temps = calc_dur_walk(testbee)
#     stopping_times, mean_stop_temps = calc_dur_stop(testbee)
#     ALL_WALKS_at_dT.extend(walking_times)
#     ALL_TEMPS_at_walk.extend(mean_walk_temps)
#     ALL_STOPS_at_dT.extend(stopping_times)
#     ALL_TEMPS_at_stop.extend(mean_stop_temps)
# fig, (subfigure_stop, subfigure_walk) = plt.subplots(1, 2, figsize=(10,5))
# fig.set_tight_layout(True)
# subfigure_stop.scatter(ALL_TEMPS_at_stop, ALL_STOPS_at_dT, color='black', alpha=0.1)
# subfigure_walk.scatter(ALL_TEMPS_at_walk, ALL_WALKS_at_dT, color='black', alpha=0.1)
# stop, stop_cov = np.polyfit(ALL_TEMPS_at_stop, ALL_STOPS_at_dT, 1, cov=True)
# walk, walk_cov = np.polyfit(ALL_TEMPS_at_walk, ALL_WALKS_at_dT, 1, cov=True)
# popt_stop, pcov_stop = curve_fit(myCurve, ALL_TEMPS_at_stop, ALL_STOPS_at_dT, maxfev = 200000, p0=(4,4))
# popt_walk, pcov_walk = curve_fit(myCurve, ALL_TEMPS_at_walk, ALL_WALKS_at_dT, maxfev = 200000, p0=(1,1))
# x = np.linspace(0,10)
# subfigure_stop.plot(x, stop[0]*x+stop[1], color='blue', linewidth=1.1)
# subfigure_stop.plot(x, myCurve(x, *popt_stop), linewidth=2, color='blue', linestyle='dashed')#, linewidth=1)
# subfigure_walk.plot(x, walk[0]*x+walk[1], color='red', linewidth=1.1)
# subfigure_walk.plot(x, myCurve(x, *popt_walk), linewidth=2, color='red', linestyle='dashed')#, linewidth=1)
# subfigure_stop.set_xlabel(r'$\Delta T \,[°C]$')
# subfigure_stop.set_ylabel("stop duration " r'$t_{s} \, [s]$')
# subfigure_walk.set_xlabel(r'$\Delta T \,[°C]$')
# subfigure_walk.set_ylabel("walk duration " r'$t_{w} \, [s]$')
# plt.show()
# """-end--------figure: STOPPING AND WALKING DURATIONS OVER TEMPERATURE-------"""

"""-begin------function: calculate stopping duration v2----------------------"""
def calc_dur_stop_2(item):
    list_vel = []
    X_dataset_wo_NAN = X_remove_NaN(item)
    Y_dataset_wo_NAN = Y_remove_NaN(item)
    n = len(X_dataset_wo_NAN) - 1
    for t in range(n):
        velocity = np.sqrt((X_dataset_wo_NAN[t + 1] - X_dataset_wo_NAN[t]) ** 2 + (Y_dataset_wo_NAN[t + 1] - Y_dataset_wo_NAN[t]) ** 2)
        list_vel.append(velocity)
    T_dataset_wo_NAN = T_remove_NaN(item)
    T_minmax_data = pd.DataFrame(data=pd.read_csv("~/Desktop/BAC_bee_mobility/final_S(pr)IN(t)_BEEs/V_3/data_bee_types/temperature_variables.csv"), columns=[item])
    T_max = T_minmax_data.iloc[1].item()
    T_min = T_minmax_data.iloc[0].item()
    stop_vel = 0.4
    duration_stop = 0
    stop_temperatures_2 = []
    for v in range(len(list_vel[:-1])):
        if list_vel[v] <= stop_vel:
            duration_stop += 1
            stop_temperatures_2.append(T_dataset_wo_NAN[v])
            if list_vel[v+1] > stop_vel:
                all_stop_durations_2.append(duration_stop)
                mean_delta_stop_temp_2.append(np.abs(np.mean(stop_temperatures_2) - T_max))
                #mean_delta_stop_temp.append(np.abs((np.mean(stop_temperatures) - T_max))/np.abs(T_max - T_min))
                duration_walk = 0
                walk_temperatures = []
            elif v == len(list_vel[:-2]):
                all_stop_durations_2.append(duration_stop)
                mean_delta_stop_temp_2.append(np.abs(np.mean(stop_temperatures_2) - T_max))
                #mean_delta_stop_temp.append(np.abs((np.mean(stop_temperatures) - T_max))/np.abs(T_max - T_min))
    return all_stop_durations_2, mean_delta_stop_temp_2
"""-end--------function: calculate stopping duration v2----------------------"""

# """-begin------figure: STOPPING OVER TEMPERATURE 2X2 GROUPED-----------------"""
# fig, ((IB1,GF1),(WF1,RW1)) = plt.subplots(2, 2, figsize=(7,7))
# n_IB=0
# fig.set_tight_layout(True)
# all_stop_durations_2 = []
# mean_delta_stop_temp_2 = []
# for item in range(0,4,1):#itemizeIB:
#     calc_dur_stop_2(Well_Behaved[n_IB])
#     n_IB += 1
# #popt_stop1, pcov_stop1 = curve_fit(myCurve1, mean_delta_stop_temp_2, all_stop_durations_2, maxfev = 200000, p0=(0,0))#, bounds=([0,0], [3.0, 1.0]))
# popt_stop2, pcov_stop2 = curve_fit(myCurve2, mean_delta_stop_temp_2, all_stop_durations_2, maxfev = 200000, p0=(0,0))#, bounds=([0,0], [3.0, 1.0]))
# #popt_stop3, pcov_stop3 = curve_fit(myCurve3, mean_delta_stop_temp, all_stop_durations, maxfev = 200000, p0=(0,0,0))#, bounds=([0,0,0], [3.0, 1.0, 1.0]))
# xspace = np.linspace(0, 10, 1000)
# IB1.set_ylim(-10,1300)
# IB1.set_xlim(-0.01,8.01)
# #IB1.plot(xspace, myCurve1(xspace, *popt_stop1), color='red', linewidth=1.1, linestyle='dashed', label=r'$ \tau / (1 + (\beta * \Delta T)) $')
# IB1.plot(xspace, myCurve2(xspace, *popt_stop2), color='red', linewidth=1.1, linestyle='dashdot', label=r'$ \tau * \exp(-\beta * \Delta T) $')
# #IB1.plot(xspace, myCurve3(xspace, *popt_stop3), color='firebrick', linewidth=1.1, label=r'$ \tau * (\beta * \Delta T) ** \gamma $')
# IB1.annotate(r'$IB_{1-4}$', xy=(0.73,0.9),xycoords='axes fraction', fontsize=12)
# IB1.scatter(mean_delta_stop_temp_2, all_stop_durations_2, color="black", alpha=0.3)
# IB1.set_ylabel(r'$t_{s}(s)$')
#
# n_GF=4
# fig.set_tight_layout(True)
# all_stop_durations_2 = []
# mean_delta_stop_temp_2 = []
# for item in range(0,4,1):#itemizeGF:
#     calc_dur_stop_2(Well_Behaved[n_GF])
#     n_GF += 1
# #popt_stop1, pcov_stop1 = curve_fit(myCurve1, mean_delta_stop_temp_2, all_stop_durations_2, maxfev = 200000, p0=(0,0))#, bounds=(0, [3.0, 1.0]))
# popt_stop2, pcov_stop2 = curve_fit(myCurve2, mean_delta_stop_temp_2, all_stop_durations_2, maxfev = 200000, p0=(0,0))#, bounds=(0, [3.0, 1.0]))
# #popt_stop3, pcov_stop3 = curve_fit(myCurve3, mean_delta_stop_temp, all_stop_durations, maxfev = 200000, p0=(0,0,0))#, bounds=(0, [3.0, 1.0, 1.0]))
# xspace = np.linspace(0, 10, 1000)
# GF1.set_ylim(-10,1300)
# GF1.set_xlim(-0.01,8.01)
# #GF1.plot(xspace, myCurve1(xspace, *popt_stop1), color='red', linewidth=1.1, linestyle='dashed', label=r'$ \tau / (1 + (\beta * \Delta T)) $')
# GF1.plot(xspace, myCurve2(xspace, *popt_stop2), color='red', linewidth=1.1, linestyle='dashdot', label=r'$ \tau * \exp(-\beta * \Delta T) $')
# #GF1.plot(xspace, myCurve3(xspace, *popt_stop3), color='firebrick', linewidth=1.1, label=r'$ \tau * (\beta * \Delta T) ** \gamma $')
# GF1.annotate(r'$GF_{1-4}$', xy=(0.73,0.9),xycoords='axes fraction', fontsize=12)
# GF1.scatter(mean_delta_stop_temp_2, all_stop_durations_2, color="black", alpha=0.3)
#
# n_WF=8
# fig.set_tight_layout(True)
# all_stop_durations_2 = []
# mean_delta_stop_temp_2 = []
# for item in range(0,4,1):#itemizeWF:
#     calc_dur_stop_2(Well_Behaved[n_WF])
#     n_WF += 1
# #popt_stop1, pcov_stop1 = curve_fit(myCurve1, mean_delta_stop_temp_2, all_stop_durations_2, maxfev = 200000, p0=(0,0))
# popt_stop2, pcov_stop2 = curve_fit(myCurve2, mean_delta_stop_temp_2, all_stop_durations_2, maxfev = 200000, p0=(0,0))
# #popt_stop3, pcov_stop3 = curve_fit(myCurve3, mean_delta_stop_temp, all_stop_durations, maxfev = 200000, p0=(0,0,0))
# xspace = np.linspace(0, 10, 1000)
# WF1.set_ylim(-10,1300)
# WF1.set_xlim(-0.01,8.01)
# #WF1.plot(xspace, myCurve1(xspace, *popt_stop1), color='red', linewidth=1.1, linestyle='dashed', label=r'$ \tau / (1 + (\beta * \Delta T)) $')
# WF1.plot(xspace, myCurve2(xspace, *popt_stop2), color='red', linewidth=1.1, linestyle='dashdot', label=r'$ \tau * e^{-\beta * \Delta T} $')
# #WF1.plot(xspace, myCurve3(xspace, *popt_stop3), color='firebrick', linewidth=1.1, label=r'$ \tau * (\beta * \Delta T) ** \gamma $')
# WF1.legend(loc="upper left")
# WF1.annotate(r'$WF_{1-4}$', xy=(0.73,0.9),xycoords='axes fraction', fontsize=12)
# WF1.scatter(mean_delta_stop_temp_2, all_stop_durations_2, color="black", alpha=0.3)
# WF1.set_xlabel(r'$\Delta T (\degree C)$')
# WF1.set_ylabel(r'$t_{s}(s)$')
#
# n_RW=12
# fig.set_tight_layout(True)
# all_stop_durations_2 = []
# mean_delta_stop_temp_2 = []
# for item in range(0,4,1):#itemizeRW:
#     calc_dur_stop_2(Well_Behaved[n_RW])
#     n_RW += 1
# #popt_stop1, pcov_stop1 = curve_fit(myCurve1, mean_delta_stop_temp_2, all_stop_durations_2, maxfev = 200000, p0=(0,0))
# popt_stop2, pcov_stop2 = curve_fit(myCurve2, mean_delta_stop_temp_2, all_stop_durations_2, maxfev = 200000, p0=(0,0))
# #popt_stop3, pcov_stop3 = curve_fit(myCurve3, mean_delta_stop_temp, all_stop_durations, maxfev = 200000, p0=(0,0,0))
# xspace = np.linspace(0, 10, 1000)
# RW1.set_ylim(-10,1300)
# RW1.set_xlim(-0.01,8.01)
# #RW1.plot(xspace, myCurve1(xspace, *popt_stop1), color='red', linewidth=1.1, linestyle='dashed', label=r'$ t_{0} = \tau / (1 + (\beta * \Delta T)) $')
# RW1.plot(xspace, myCurve2(xspace, *popt_stop2), color='red', linewidth=1.1, linestyle='dashdot', label=r'$ t_{0} = \tau * e^{-\beta * \Delta T} $')
# #RW1.plot(xspace, myCurve3(xspace, *popt_stop3), color='firebrick', linewidth=1.1, label=r'$ t_{0} = \tau * (\beta * \Delta T)^{\gamma} $')
# RW1.annotate(r'$RW_{1-4}$', xy=(0.73,0.9),xycoords='axes fraction', fontsize=12)
# RW1.scatter(mean_delta_stop_temp_2, all_stop_durations_2, color="black", alpha=0.3)
# RW1.set_xlabel(r'$\Delta T (\degree C)$')
#
# plt.show()
# """-end--------figure: STOPPING OVER TEMPERATURE 2X2 GROUPED-----------------"""

# """-begin--------figure: STOPPING OVER TEMPERATURE 16 BEES-------------------"""
# fig, ((IB1, IB2, IB3, IB4), (GF1, GF2, GF3, GF4), (WF1, WF2, WF3, WF4), (RW1, RW2, RW3, RW4)) = plt.subplots(4, 4, figsize=(10,10))
# itemize = [IB1, IB2, IB3, IB4, GF1, GF2, GF3, GF4, WF1, WF2, WF3, WF4, RW1, RW2, RW3, RW4]
# type = [r'$IB_1$', r'$IB_2$', r'$IB_3$', r'$IB_4$', r'$GF_1$', r'$GF_2$', r'$GF_3$', r'$GF_4$', r'$WF_1$', r'$WF_2$', r'$WF_3$', r'$WF_4$', r'$RW_1$', r'$RW_2$', r'$RW_3$', r'$RW_4$']
# n=0
# fig.set_tight_layout(True)
# for item in itemize:
#     item.set_ylim(-20,1300)
#     item.set_xlim(-0.2,10.2)
#     xspace = np.linspace(0,10,99)
#     vel = calc_vel(Well_Behaved[n])
#     vel_0 = np.mean(vel[0])
#     stop_durations, stop_temp = calc_dur_stop(Well_Behaved[n])
#     item.scatter(stop_temp, stop_durations, color='black', alpha=0.3)
#     item.annotate(type[n], xy=(0.8,0.9),xycoords='axes fraction', fontsize=12)
#     item.set_xticks([])
#     item.set_yticks([])
#     if item != IB2:
#         # popt1, pcov1 = curve_fit(myCurve1, stop_temp, stop_durations, maxfev = 200000, p0=(0,0))
#         # item.plot(myCurve1(xspace, *popt1), color="red", linestyle='dashdot', label='t/(1+bx)')
#         popt2, pcov2 = curve_fit(myCurve2, stop_temp, stop_durations, maxfev = 200000, p0=(0,0))
#         item.plot(myCurve2(np.linspace(0,10,11), *popt2), color="red", linestyle='dotted', label='t*e^(-bx)')
#         # popt3, pcov3 = curve_fit(myCurve3, stop_temp, stop_durations, maxfev = 200000, p0=(0,0,0))
#         # item.plot(myCurve3(np.linspace(0,100,100), *popt3), color="red", linestyle='dashed', label='t*(bx)^g')
#     if item == IB2:
#         item.annotate('invalid', xy=(0.35,0.45),xycoords='axes fraction', color='red', fontsize=12)
#     if item in [RW1, RW2, RW3, RW4]:
#         item.set_xticks([0,5,10])
#         item.set_xlabel(r'$\Delta T (\degree C)$')
#     if item in [IB1, GF1, WF1, RW1]:
#         item.set_yticks([0,600,1200])
#         item.set_ylabel(r'$t_{s}(s)$')
#     n += 1
# plt.show()
# """-end----------figure: STOPPING OVER TEMPERATURE 16 BEES-------------------"""

"""-begin------function: calculate probability at a temperature--------------"""
def probability_T(item):
    T_dataset_wo_NAN = T_remove_NaN(item)
    T_minmax_data = pd.DataFrame(data=pd.read_csv("data_bee_types/temperature_variables.csv"), columns=[item])
    T_max = T_minmax_data.iloc[1].item()
    prob_dataset = []
    for T in range(len(T_dataset_wo_NAN)):
        prob_dataset.append(np.abs(T_dataset_wo_NAN[T] - T_max))
    return prob_dataset
"""-end--------function: calculate probability at a temperature--------------"""

"""-begin------function: calculate velocity at stopping events---------------"""
def calc_vel_at_noonly_stopping(item):
    list_vel = []
    X_dataset_wo_NAN = X_remove_NaN(item)
    Y_dataset_wo_NAN = Y_remove_NaN(item)
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
"""-end--------function: calculate velocity at stopping events---------------"""

# """-begin--------figure: PSD OF RTS 16 BEES----------------------------------"""
# fig, ((IB1, IB2, IB3, IB4), (GF1, GF2, GF3, GF4), (WF1, WF2, WF3, WF4), (RW1, RW2, RW3, RW4)) = plt.subplots(4, 4, figsize=(10,10))
# fig.set_tight_layout(True)
# itemize = [IB1, IB2, IB3, IB4, GF1, GF2, GF3, GF4, WF1, WF2, WF3, WF4, RW1, RW2, RW3, RW4]
# type = [r'$IB_1$', r'$IB_2$', r'$IB_3$', r'$IB_4$', r'$GF_1$', r'$GF_2$', r'$GF_3$', r'$GF_4$', r'$WF_1$', r'$WF_2$', r'$WF_3$', r'$WF_4$', r'$RW_1$', r'$RW_2$', r'$RW_3$', r'$RW_4$']
# n = 0
# start = 0
# end = 6
# dT = 0.1
# xspace = np.linspace(0, 10, 1000)
# for item in itemize:
#     item.annotate(type[n], xy=(0.8,0.9),xycoords='axes fraction', fontsize=12)
#     if item != IB2:
#         vel_PSD = PSD_vel(Well_Behaved[n])
#         length = len(vel_PSD)
#         start_new = 0.001
#         end_new = 2 * np.pi / 2
#         omega_new = np.linspace(start_new, end_new, length)
#         item.plot(omega_new, vel_PSD, color='black', alpha=0.5)
#         mean_velocity = np.mean(calc_vel_at_noonly_stopping(Well_Behaved[n])[0])
#         all_stop_durations, mean_delta_stop_temp = calc_dur_stop(Well_Behaved[n])
#         all_walk_durations, mean_delta_walk_temp = calc_dur_walk(Well_Behaved[n])
#         mean_t1 = np.mean(all_walk_durations)
#         #popt_stop1, pcov_stop1 = curve_fit(myCurve1, mean_delta_stop_temp, all_stop_durations, maxfev = 200000, p0=(0,0))#, bounds=([0,0], [3.0, 1.0]))
#         popt_stop2, pcov_stop2 = curve_fit(myCurve2, mean_delta_stop_temp, all_stop_durations, maxfev = 200000, p0=(0,0))#, bounds=([0,0], [3.0, 1.0]))
#         #popt_stop3, pcov_stop3 = curve_fit(myCurve3, mean_delta_stop_temp, all_stop_durations, maxfev = 200000, p0=(0,0,0))#, bounds=([0,0,0], [3.0, 1.0, 1.0]))
#         prob_dataset = probability_T(Well_Behaved[n])
#         bins = np.arange(start, end, dT)
#         rangeT = np.linspace(start, end, 59)
#         data_entries, bins = np.histogram(prob_dataset, bins=bins, range=(start,end), density=True)
#         def S(w, v_0, t_1, Prob, dTemp):
#             sum = 0
#             for T in np.arange(0,5.9,0.1):
#                 sum += (Prob[int(round(T/0.1, 0))] * dTemp) / ((myCurve2(T, *popt_stop2) + 1) * ((1 / myCurve2(T, *popt_stop2) + 1 / t_1) ** 2 + w ** 2))
#             return 2 * v_0 ** 2 * sum
#         omega = np.linspace(0.0001, (np.pi/2))
#         item.plot(omega, S(omega, mean_velocity, mean_t1, data_entries, dT), linestyle='dashed', color='black')
#         x_low = 0.1
#         x_high = 1.0
#         y_low = S(x_low, mean_velocity, mean_t1, data_entries, dT)
#         y_high = S(x_high, mean_velocity, mean_t1, data_entries, dT)
#         item.plot(x_low, y_low , marker="^", color='red')
#         item.plot(x_high, y_high, marker="^", color='red')
#         alpha = np.abs(np.log10(y_high/y_low))
#         item.plot(omega, powerCurve(omega, alpha), linestyle='dotted', color='red')
#         item.annotate(r'$\alpha_{1} = $' + str(round(alpha,2)) , xy=(0.05,0.9),xycoords='axes fraction', fontsize=12)
#
#         #high_omega = np.linspace(0.15,1.5)
#         high_omega = np.linspace(0.015,1.5)
#         popt_pow, pcov_pow = curve_fit(powerCurve, high_omega, S(high_omega, mean_velocity, mean_t1, data_entries, dT), maxfev=200000, p0=(0))
#         item.plot(omega, powerCurve(high_omega, *popt_pow), linestyle='dashed', color='red')
#         item.annotate(r'$\alpha_{2} = $' + str(round(popt_pow[0],2)) , xy=(0.05,0.8),xycoords='axes fraction', fontsize=12)
#     else:
#         item.annotate('invalid', xy=(0.35,0.45),xycoords='axes fraction', color='red', fontsize=12)
#         item.plot(omega_new, vel_PSD, color='black', alpha=0.5)
#     item.set_xscale('log')
#     item.set_yscale('log')
#     item.set_xlim(10 ** (-3), (np.pi/2) * 10 ** 0)
#     item.set_ylim(10 ** (-4), 10 ** 4)
#     item.set_xticks([])
#     item.set_yticks([])
#     if item in [IB1, GF1, WF1, RW1]:
#         item.set_ylabel(r'$S_{RTS}(\omega)$')
#         item.set_yticks([10 ** (-2), 10**(0), 10 ** (2)])
#     if item in [RW1, RW2, RW3, RW4]:
#         item.set_xlabel(r'$\omega$')
#         item.set_xticks([10 ** (-2), 10**(-1), 10 ** 0])
#     n += 1
# plt.show()
# """-end----------figure: PSD OF RTS 16 BEES----------------------------------"""

"""-begin------function: calculate gradient----------------------------------"""
def calc_delta_T(item):
    calc_dataset = T_remove_NaN(item)
    delta_T = []
    for i in range(len(calc_dataset) - 1):
        delta_T.append(calc_dataset[i + 1] - calc_dataset[i])
    return delta_T
"""-end--------function: calculate gradient----------------------------------"""

# """-begin------figure: MEAN BIN VELOCITY OVER DELTA TEMPERATURE ALL BEES-----"""
# T_master_bin = np.arange(27, 37, 0.1)
# v_mean_list = np.zeros(100)
# m_list = np.zeros(100)
# for testbee in head:
#     vel = calc_vel(testbee)
#     T = T_remove_NaN(testbee)
#     plt.scatter(T, vel, color='black',  alpha=0.01)
#     T_bin = np.arange(min(T), max(T), 0.1)
#     bin_vel_LIST = []
#     for i in T_bin:
#         bin_vel = 0
#         n = 0.000001
#         for ii in range(len(vel)):
#             if np.abs(T[ii] - i) < 0.1:
#                 bin_vel += vel[ii]
#                 n += 1
#         bin_vel_LIST.append(bin_vel/n)
#     for j in range(len(T_master_bin)):
#         for jj in range(len(T_bin)):
#             if np.abs(T_master_bin[j]-T_bin[jj]) < 0.1:
#                 v_mean_list[j] += bin_vel_LIST[jj]
#                 m_list[j] += 1
# for v in range(len(v_mean_list)):
#     v_mean_list[v] /= m_list[v]
# plt.plot(T_master_bin, v_mean_list, label="all experiments", color="slategrey", linewidth=2)
#
# T_master_bin = np.arange(27, 37, 0.1)
# v_mean_list = np.zeros(100)
# m_list = np.zeros(100)
# for testbee in NARROW:
#     vel = calc_vel(testbee)
#     T = T_remove_NaN(testbee)
#     T_bin = np.arange(min(T), max(T), 0.1)
#     bin_vel_LIST = []
#     for i in T_bin:
#         bin_vel = 0
#         n = 0.000001
#         for ii in range(len(vel)):
#             if np.abs(T[ii] - i) < 0.1:
#                 bin_vel += vel[ii]
#                 n += 1
#         bin_vel_LIST.append(bin_vel/n)
#     for j in range(len(T_master_bin)):
#         for jj in range(len(T_bin)):
#             if np.abs(T_master_bin[j]-T_bin[jj]) < 0.1:
#                 v_mean_list[j] += bin_vel_LIST[jj]
#                 m_list[j] += 1
# for v in range(len(v_mean_list)):
#     v_mean_list[v] /= m_list[v]
# plt.plot(T_master_bin, v_mean_list, label="narrow gradient", color="black", linestyle="dotted")
#
# T_master_bin = np.arange(27, 37, 0.1)
# v_mean_list = np.zeros(100)
# m_list = np.zeros(100)
# for testbee in STEEP:
#     vel = calc_vel(testbee)
#     T = T_remove_NaN(testbee)
#     T_bin = np.arange(min(T), max(T), 0.1)
#     bin_vel_LIST = []
#     for i in T_bin:
#         bin_vel = 0
#         n = 0.000001
#         for ii in range(len(vel)):
#             if np.abs(T[ii] - i) < 0.1:
#                 bin_vel += vel[ii]
#                 n += 1
#         bin_vel_LIST.append(bin_vel/n)
#     for j in range(len(T_master_bin)):
#         for jj in range(len(T_bin)):
#             if np.abs(T_master_bin[j]-T_bin[jj]) < 0.1:
#                 v_mean_list[j] += bin_vel_LIST[jj]
#                 m_list[j] += 1
# for v in range(len(v_mean_list)):
#     v_mean_list[v] /= m_list[v]
# plt.plot(T_master_bin, v_mean_list, label="steep gradient", color="black", linestyle="dashed")
#
# T_master_bin = np.arange(27, 37, 0.1)
# v_mean_list = np.zeros(100)
# m_list = np.zeros(100)
# for testbee in STEEPEST:
#     vel = calc_vel(testbee)
#     T = T_remove_NaN(testbee)
#     T_bin = np.arange(min(T), max(T), 0.1)
#     bin_vel_LIST = []
#     for i in T_bin:
#         bin_vel = 0
#         n = 0.000001
#         for ii in range(len(vel)):
#             if np.abs(T[ii] - i) < 0.1:
#                 bin_vel += vel[ii]
#                 n += 1
#         bin_vel_LIST.append(bin_vel/n)
#     for j in range(len(T_master_bin)):
#         for jj in range(len(T_bin)):
#             if np.abs(T_master_bin[j]-T_bin[jj]) < 0.1:
#                 v_mean_list[j] += bin_vel_LIST[jj]
#                 m_list[j] += 1
# for v in range(len(v_mean_list)):
#     v_mean_list[v] /= m_list[v]
# plt.plot(T_master_bin, v_mean_list, label="steepest gradient", color="black", linestyle="dashdot")
#
# T_master_bin = np.arange(27, 37, 0.1)
# v_mean_list = np.zeros(100)
# m_list = np.zeros(100)
# for testbee in control_30_30:
#     vel = calc_vel(testbee)
#     T = T_remove_NaN(testbee)
#     T_bin = np.arange(min(T), max(T), 0.1)
#     bin_vel_LIST = []
#     for i in T_bin:
#         bin_vel = 0
#         n = 0.000001
#         for ii in range(len(vel)):
#             if np.abs(T[ii] - i) < 0.1:
#                 bin_vel += vel[ii]
#                 n += 1
#         bin_vel_LIST.append(bin_vel/n)
#     for j in range(len(T_master_bin)):
#         for jj in range(len(T_bin)):
#             if np.abs(T_master_bin[j]-T_bin[jj]) < 0.1:
#                 v_mean_list[j] += bin_vel_LIST[jj]
#                 m_list[j] += 1
# for v in range(len(v_mean_list)):
#     v_mean_list[v] /= m_list[v]
# plt.plot(T_master_bin, v_mean_list, label="control (30-30)", color="steelblue", linestyle=(0, (3, 1, 1, 1, 1, 1)))
#
# T_master_bin = np.arange(27, 37, 0.1)
# v_mean_list = np.zeros(100)
# m_list = np.zeros(100)
# for testbee in control_36_36:
#     vel = calc_vel(testbee)
#     T = T_remove_NaN(testbee)
#     T_bin = np.arange(min(T), max(T), 0.1)
#     bin_vel_LIST = []
#     for i in T_bin:
#         bin_vel = 0
#         n = 0.000001
#         for ii in range(len(vel)):
#             if np.abs(T[ii] - i) < 0.1:
#                 bin_vel += vel[ii]
#                 n += 1
#         bin_vel_LIST.append(bin_vel/n)
#     for j in range(len(T_master_bin)):
#         for jj in range(len(T_bin)):
#             if np.abs(T_master_bin[j]-T_bin[jj]) < 0.1:
#                 v_mean_list[j] += bin_vel_LIST[jj]
#                 m_list[j] += 1
# for v in range(len(v_mean_list)):
#     v_mean_list[v] /= m_list[v]
# plt.plot(T_master_bin, v_mean_list, label="control (36-36)", color="red", linestyle=(0, (3, 1, 1, 1)))
# plt.xlim(28,36)
#
# plt.ylim(0,4)
# plt.xlabel('Temperature[°C]')
# plt.ylabel('velocity[cm/s]')
# plt.legend(loc="upper right")
# plt.show()
# """-begin------figure: MEAN BIN VELOCITY OVER DELTA TEMPERATURE ALL BEES-----"""
