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
    for tau in range(1, length):
        MSD_tau = 0
        for t in range(0, (length-tau)):
            MSD_tau += (x[t+tau]-x[t])**2 + (y[t+tau]-y[t])**2
        MSD.append(MSD_tau/(length-tau))
    return MSD

def myExpFunc(x, b, m):
    return x * m + b #x ** m + b

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
