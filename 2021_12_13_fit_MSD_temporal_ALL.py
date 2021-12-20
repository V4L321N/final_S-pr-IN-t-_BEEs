import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import curve_fit

"""-sample figure MSD-"""
def diffusion():
    time = np.linspace(0,100,100)
    alpha_sub = 0.7
    alpha = 1
    alpha_sup = 1.3
    list_diff_sub = []
    list_diff = []
    list_diff_sup = []
    for i in range(len(time)):
        list_diff_sub.append((i) ** alpha_sub)
        list_diff.append((i) ** alpha)
        list_diff_sup.append(i ** alpha_sup)
    return time, list_diff_sub, list_diff, list_diff_sup
x,sub,diff,sup = diffusion()
# fig, ax = plt.subplots(figsize=(5,5))
# ax.plot(x,sub, 'black', linewidth=2, linestyle='dotted', label="sub-diffusive")
# ax.plot(x,diff, 'black', linewidth=2, label="normal-diffusive")
# ax.plot(x,sup, 'black', linewidth=2, linestyle='dashdot', label="super-diffusive")
# ax.set_xlim(left=0, right=10)
# ax.set_ylim(bottom=0, top=10)
# ax.set_xticks(())
# ax.set_yticks(())
# plt.show()
#----------------------------------
# plt.figure(figsize=(4.5,4.5))
# x,sub,diff,sup = diffusion()
# plt.xlim(1,10)
# plt.ylim(1,10)
# plt.xticks(())
# plt.yticks(())
# plt.plot(x,sub,color='black',linestyle='dotted', label="sub-diffusive")
# plt.plot(x,diff,color='black', label="normal-diffusive")
# plt.plot(x,sup,color='black',linestyle='dashdot', label="super-diffusive")
# plt.xlabel(r'$\tau$')
# plt.ylabel(r'$D$')
# plt.legend(loc='lower right')
# plt.savefig("Figures/MSD_schematic")
# plt.show()

"""---"""

head = list(pd.DataFrame(data=pd.read_csv("data_bee_types/LS_spatial_D_x.csv")))

control_30_30 = ["BT17A-1", "BT17A-2", "BT17A-3", "BT17A-4", "BT17B-1", "BT17B-2", "BT17B-3", "BT17B-4"]
control_36_36 = ["BT07A-1", "BT07A-2", "BT07A-3", "BT07A-4", "BT07B-1", "BT07B-2", "BT07B-3", "BT07B-4"]
control_ALL = control_30_30 + control_36_36
for item in control_ALL:
    head.remove(item)

bad_bees = ["BT11A-1","BT11A-4", "BT01A-1","BT02B-1", "BT03A-1", "BT03A-2", "BT03A-3", "BT03A-4", "BT03B-1", "BT03B-2", "BT03B-3", "BT03B-4"]
# #BT11A-1 and "BT11A-4 needs to be removed from the bad bees list.
for item in bad_bees:
    head.remove(item)

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

testbee = "BT08A-3"

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


x,y = X_remove_NaN(testbee),Y_remove_NaN(testbee)

"""plot trajectory"""
# plt.plot(x,y)
# circle1 = plt.Circle((30, 30), 30, linestyle='dashed', color='black', fill=False)
# plt.gca().add_patch(circle1)
# plt.xlim(0,60)
# plt.ylim(0,60)
# plt.show()
"""-------------"""


"""--MSD bundle-----"""
# def MSD_bundle(item):
#     lin = np.linspace(0,500,10)
#     x,y = X_remove_NaN(item),Y_remove_NaN(item)
#     posx = 0
#     posy = 0
#     MSD = [0]
#     slope = []
#     for i in range(1,len(x)):
#         if np.sqrt((x[i]-30)**2+(y[i]-30)**2) > 29:
#             posx = 0
#             posy = 0
#             plt.plot(MSD)
#             slope.append(MSD[-1]/len(MSD))
#             MSD = [0]
#         else:
#             MSD.append(np.sqrt(posx**2 + posy**2))
#             posx += np.sqrt((x[i-1]-x[i])**2)
#             posy += np.sqrt((y[i-1]-y[i])**2)
#     slope.append(MSD[-1]/len(MSD))
#     print(slope)
#     plt.plot(MSD)
#     plt.plot(lin,lin,linestyle='dotted',color='black')
#     #plt.xlim(0,300)
#     #plt.ylim(0,800)
#     #plt.xscale("log")
#     #plt.yscale("log")
#     plt.title(item)
#     plt.show()
#
# for bee in Well_Behaved:
#     MSD_bundle(bee)
#
"""-------------"""

"""---MSD temporal mean---"""
#
# def MSD_temporal(item):
#     x,y = X_remove_NaN(item),Y_remove_NaN(item)
#     MSD = []
#     for tau in range(1,math.floor(len(x)/2)):
#         MSD_tau = 0
#         for t in range(tau):
#             MSD_tau += (x[t+tau]-x[t])**2 + (y[t+tau]-y[t])**2
#         MSD.append((1/tau)*MSD_tau)
#     plt.plot(MSD)
#     plt.xlabel(r'$\tau$')
#     plt.ylabel(r'$\langle r^{2}(\tau) \rangle$')
#     plt.title(item)
#     plt.xscale("log")
#     plt.yscale("log")
#     plt.show()

def MSD_temporal(item):
    x,y = X_remove_NaN(item),Y_remove_NaN(item)
    MSD = []
    length = len(x)
    for tau in range(0, length):
        MSD_tau = 0
        for t in range(0, (length-tau)):
            MSD_tau += (x[t+tau]-x[t])**2 + (y[t+tau]-y[t])**2
        MSD.append(MSD_tau/(length-tau))
    return MSD

def myExpFunc(x, k1, k2):
    return x ** k1 + k2

def myLinFunc(x, b, m):
    return m * x + b

def myIntercept(x,c1,c2,c3):
    return c3 * x ** (c1) + c2 #np.exp(c2)

# def myIntercept(x,c1,c2):
#     return c1 * x ** (2) + c2 #np.exp(c2)
inter_list = []
exp_list = []
for item in head:
    listy = MSD_temporal(item)
    listx = np.linspace(0, len(listy), len(listy))
    plt.plot(listx, listy, color='black', alpha=0.5, linewidth=0.3)
    inter_list.append(listy[1])

    # START_inter = 20
    # END_inter = len(listy)
    # listyINTER = listy[START_inter:END_inter]
    # listxINTER = np.linspace(START_inter, END_inter, END_inter-START_inter)
    # popt_inter, pcov_inter = curve_fit(myIntercept, listxINTER, listyINTER, maxfev = 2000000, p0=(1, 350, 1))
    # plt.plot(listx, myIntercept(listx, *popt_inter), label='fit', color='red', linestyle='dashdot')
    # plt.plot(listxINTER, listyINTER, label='MSD', color='red', alpha=0.7)
    #coefficients = np.polyfit(np.log10(listxINTER), np.log10(listyINTER), 1)
    #polynomial = np.poly1d(coefficients)
    #log10_y_fit = polynomial(np.log10(x))
    #plt.plot(x, 10**log10_y_fit, '-')
    #
    START_lin = 0
    END_lin = 20 #len(listy)
    listyLIN = listy[START_lin:END_lin]
    listxLIN = np.linspace(START_lin, END_lin, END_lin-START_lin)
    popt_lin, pcov_lin = curve_fit(myExpFunc, listxLIN, listyLIN, maxfev = 200000, p0=(2, 30))
    exp_list.append(popt_lin[0])
    #plt.plot(listxLIN, myExpFunc(listxLIN, *popt_lin), alpha=0.3, linewidth=0.5, color='blue', linestyle='dotted')#, linewidth=1)
    #plt.plot(listxLIN, listyLIN, label='MSD', color='blue', alpha=0.7)

    # popt, pcov = curve_fit(myExpFunc, listxINTER, listyINTER)
    # print(popt, pcov)
    # plt.plot(listx, myExpFunc(listx, *popt), label='fit', color='black', linestyle='dotted')


    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(1,1200)
    #plt.ylim(0,4000)
    #plt.show()
    #print(popt_inter)
#plt.xticks([])
#plt.yticks([])
#item.text(400, 3500, r'$m= $' + str(round(popt[1], 3)))
#item.text(400, 3100,r'$b= $' + str(round(popt[0], 3)))
std_inter = np.std(inter_list)
mean_inter = np.mean(inter_list)
mean_exp = np.mean(exp_list)
plt.plot(listx, (listx/listx) * mean_inter, color='red', alpha=0.8, linewidth=0.9, linestyle='dotted')
plt.plot(listx, listx**1.8, color='blue', alpha=0.8, linewidth=0.9)
plt.xlabel(r'$\tau$')
plt.ylabel(r'$\langle r^{2}(\tau) \rangle$')
plt.show()

"""----------------------"""


"""----try 1------"""
# posx = 0
# posy = 0
# trajectoryx = []
# trajectoryy = []
# for i in range(len(x)):
#     if np.sqrt((x[i]-30)**2+(y[i]-30)**2) > 29:
#         posx = 0
#         posy = 0
#         #trajectoryx = []
#         #trajectoryy = []
#     else:
#         posx += x[i]-x[i-1]
#         posy += y[i]-y[i-1]
#     trajectoryx.append(posx)
#     trajectoryy.append(posy)
# plt.plot(trajectoryx,trajectoryy)
# plt.show()
"""-------------"""


"""-----try 2--------"""
# def MSD_sum(item):
#     MSD_sum_list = []
#     MSD = 0
#     x,y = X_remove_NaN(item),Y_remove_NaN(item)
#     for i in range(len(x)):
#
#         MSD += np.sqrt((x[i]-x[i-1])**2 + (y[i]-y[i-1])**2)
#         MSD_sum_list.append(MSD)
#     plt.plot(MSD_sum_list)
# for bee in head:
#     MSD_sum(bee)
#     plt.show()
"""-------------"""
