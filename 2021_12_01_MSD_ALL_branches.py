import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

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
Immobile_Bees = [IB_1, IB_2, IB_3, IB_4]
Goal_Finders = [GF_1, GF_2, GF_3, GF_4]
Wall_Followers = [WF_1, WF_2, WF_3, WF_4]
Random_Walkers = [RW_1, RW_2, RW_3, RW_4]
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


"""--MSD IB bundle-----"""
def MSD_bundle(item):
    lin = np.linspace(0,500,10)
    x,y = X_remove_NaN(item),Y_remove_NaN(item)
    posx = 0
    posy = 0
    MSD = [0]
    slope = []

    if item == "BT06A-2":
        for i in range(1,len(x)):
            if np.sqrt((x[i]-30)**2+(y[i]-30)**2) > 29:
                posx = 0
                posy = 0
                IB1.plot(MSD, color='black', alpha=0.3)
                slope.append(MSD[-1]/len(MSD))
                MSD = [0]
            else:
                MSD.append(np.sqrt(posx**2 + posy**2))
                posx += np.sqrt((x[i-1]-x[i])**2)
                posy += np.sqrt((y[i-1]-y[i])**2)
        slope.append(MSD[-1]/len(MSD))
        IB1.plot(MSD, color='black', alpha=0.3)
        IB1.plot(lin,lin,linestyle='dotted',color='black')
        #IB1.set_xlabel(r'$\tau$')
        #IB1.set_xticklabels([])
        IB1.set_ylabel(r'$\langle r^{2}(\tau) \rangle$')
        IB1.set_xticks([500, 1000])
        IB1.title.set_text(r'$IB_{1}$')


    elif item == "BT06A-3":
        for i in range(1,len(x)):
            if np.sqrt((x[i]-30)**2+(y[i]-30)**2) > 29:
                posx = 0
                posy = 0
                IB2.plot(MSD, color='black', alpha=0.3)
                slope.append(MSD[-1]/len(MSD))
                MSD = [0]
            else:
                MSD.append(np.sqrt(posx**2 + posy**2))
                posx += np.sqrt((x[i-1]-x[i])**2)
                posy += np.sqrt((y[i-1]-y[i])**2)
        slope.append(MSD[-1]/len(MSD))
        IB2.plot(MSD, color='black', alpha=0.3)
        IB2.plot(lin,lin,linestyle='dotted',color='black')
        IB2.set_yticklabels([])
        #IB2.set_xticklabels([])
        #IB2.set_xlabel(r'$\tau$')
        IB2.set_xticks([500, 1000])
        IB2.title.set_text(r'$IB_{2}$')

    elif item == "BT02B-2":
        for i in range(1,len(x)):
            if np.sqrt((x[i]-30)**2+(y[i]-30)**2) > 29:
                posx = 0
                posy = 0
                IB3.plot(MSD, color='black', alpha=0.3)
                slope.append(MSD[-1]/len(MSD))
                MSD = [0]
            else:
                MSD.append(np.sqrt(posx**2 + posy**2))
                posx += np.sqrt((x[i-1]-x[i])**2)
                posy += np.sqrt((y[i-1]-y[i])**2)
        slope.append(MSD[-1]/len(MSD))
        IB3.plot(MSD, color='black', alpha=0.3)
        IB3.plot(lin,lin,linestyle='dotted',color='black')
        IB3.set_yticklabels([])
        #IB3.set_xticklabels([])
        #IB3.set_xlabel(r'$\tau$')
        IB3.set_xticks([500, 1000])
        IB3.title.set_text(r'$IB_{3}$')

    elif item == "BT02B-1":
        for i in range(1,len(x)):
            if np.sqrt((x[i]-30)**2+(y[i]-30)**2) > 29:
                posx = 0
                posy = 0
                IB4.plot(MSD, color='black', alpha=0.3)
                slope.append(MSD[-1]/len(MSD))
                MSD = [0]
            else:
                MSD.append(np.sqrt(posx**2 + posy**2))
                posx += np.sqrt((x[i-1]-x[i])**2)
                posy += np.sqrt((y[i-1]-y[i])**2)
        slope.append(MSD[-1]/len(MSD))
        IB4.plot(MSD, color='black', alpha=0.3)
        IB4.plot(lin,lin,linestyle='dotted',color='black')
        IB4.set_yticklabels([])
        #IB4.set_xticklabels([])
        #IB4.set_xlabel(r'$\tau$')
        IB4.set_xticks([500, 1000])
        IB4.title.set_text(r'$IB_{4}$')

    elif item == "BT09A-2":
        for i in range(1,len(x)):
            if np.sqrt((x[i]-30)**2+(y[i]-30)**2) > 29:
                posx = 0
                posy = 0
                GF1.plot(MSD, color='black', alpha=0.3)
                slope.append(MSD[-1]/len(MSD))
                MSD = [0]
            else:
                MSD.append(np.sqrt(posx**2 + posy**2))
                posx += np.sqrt((x[i-1]-x[i])**2)
                posy += np.sqrt((y[i-1]-y[i])**2)
        slope.append(MSD[-1]/len(MSD))
        GF1.plot(MSD, color='black', alpha=0.3)
        GF1.plot(lin,lin,linestyle='dotted',color='black')
        #GF1.set_xlabel(r'$\tau$')
        #GF1.set_xticklabels([])
        GF1.set_ylabel(r'$\langle r^{2}(\tau) \rangle$')
        GF1.set_xticks([500, 1000])
        GF1.title.set_text(r'$GF_{1}$')

    elif item == "BT09B-2":
        for i in range(1,len(x)):
            if np.sqrt((x[i]-30)**2+(y[i]-30)**2) > 29:
                posx = 0
                posy = 0
                GF2.plot(MSD, color='black', alpha=0.3)
                slope.append(MSD[-1]/len(MSD))
                MSD = [0]
            else:
                MSD.append(np.sqrt(posx**2 + posy**2))
                posx += np.sqrt((x[i-1]-x[i])**2)
                posy += np.sqrt((y[i-1]-y[i])**2)
        slope.append(MSD[-1]/len(MSD))
        GF2.plot(MSD, color='black', alpha=0.3)
        GF2.plot(lin,lin,linestyle='dotted',color='black')
        GF2.set_yticklabels([])
        #GF2.set_xticklabels([])
        #GF2.set_xlabel(r'$\tau$')
        GF2.set_xticks([500, 1000])
        GF2.title.set_text(r'$GF_{2}$')

    elif item == "BT09B-4":
        for i in range(1,len(x)):
            if np.sqrt((x[i]-30)**2+(y[i]-30)**2) > 29:
                posx = 0
                posy = 0
                GF3.plot(MSD, color='black', alpha=0.3)
                slope.append(MSD[-1]/len(MSD))
                MSD = [0]
            else:
                MSD.append(np.sqrt(posx**2 + posy**2))
                posx += np.sqrt((x[i-1]-x[i])**2)
                posy += np.sqrt((y[i-1]-y[i])**2)
        slope.append(MSD[-1]/len(MSD))
        GF3.plot(MSD, color='black', alpha=0.3)
        GF3.plot(lin,lin,linestyle='dotted',color='black')
        GF3.set_yticklabels([])
        #GF3.set_xticklabels([])
        #GF3.set_xlabel(r'$\tau$')
        GF3.set_xticks([500, 1000])
        GF3.title.set_text(r'$GF_{3}$')

    elif item == "BT12B-2":
        for i in range(1,len(x)):
            if np.sqrt((x[i]-30)**2+(y[i]-30)**2) > 29:
                posx = 0
                posy = 0
                GF4.plot(MSD, color='black', alpha=0.3)
                slope.append(MSD[-1]/len(MSD))
                MSD = [0]
            else:
                MSD.append(np.sqrt(posx**2 + posy**2))
                posx += np.sqrt((x[i-1]-x[i])**2)
                posy += np.sqrt((y[i-1]-y[i])**2)
        slope.append(MSD[-1]/len(MSD))
        GF4.plot(MSD, color='black', alpha=0.3)
        GF4.plot(lin,lin,linestyle='dotted',color='black')
        GF4.set_yticklabels([])
        #GF4.set_xticklabels([])
        #GF4.set_xlabel(r'$\tau$')
        GF4.set_xticks([500, 1000])
        GF4.title.set_text(r'$GF_{4}$')

    elif item == "BT04B-3":
        for i in range(1,len(x)):
            if np.sqrt((x[i]-30)**2+(y[i]-30)**2) > 29:
                posx = 0
                posy = 0
                WF1.plot(MSD, color='black', alpha=0.3)
                slope.append(MSD[-1]/len(MSD))
                MSD = [0]
            else:
                MSD.append(np.sqrt(posx**2 + posy**2))
                posx += np.sqrt((x[i-1]-x[i])**2)
                posy += np.sqrt((y[i-1]-y[i])**2)
        slope.append(MSD[-1]/len(MSD))
        WF1.plot(MSD, color='black', alpha=0.3)
        WF1.plot(lin,lin,linestyle='dotted',color='black')
        #WF1.set_xticklabels([])
        #WF1.set_xlabel(r'$\tau$')
        WF1.set_ylabel(r'$\langle r^{2}(\tau) \rangle$')
        WF1.set_xticks([200, 400])
        WF1.title.set_text(r'$WF_{1}$')

    elif item == "BT09B-1":
        for i in range(1,len(x)):
            if np.sqrt((x[i]-30)**2+(y[i]-30)**2) > 29:
                posx = 0
                posy = 0
                WF2.plot(MSD, color='black', alpha=0.3)
                slope.append(MSD[-1]/len(MSD))
                MSD = [0]
            else:
                MSD.append(np.sqrt(posx**2 + posy**2))
                posx += np.sqrt((x[i-1]-x[i])**2)
                posy += np.sqrt((y[i-1]-y[i])**2)
        slope.append(MSD[-1]/len(MSD))
        WF2.plot(MSD, color='black', alpha=0.3)
        WF2.plot(lin,lin,linestyle='dotted',color='black')
        WF2.set_yticklabels([])
        #WF2.set_xticklabels([])
        #WF2.set_xlabel(r'$\tau$')
        WF2.set_xticks([200, 400])
        WF2.title.set_text(r'$WF_{2}$')

    elif item == "BT12B-1":
        for i in range(1,len(x)):
            if np.sqrt((x[i]-30)**2+(y[i]-30)**2) > 29:
                posx = 0
                posy = 0
                WF3.plot(MSD, color='black', alpha=0.3)
                slope.append(MSD[-1]/len(MSD))
                MSD = [0]
            else:
                MSD.append(np.sqrt(posx**2 + posy**2))
                posx += np.sqrt((x[i-1]-x[i])**2)
                posy += np.sqrt((y[i-1]-y[i])**2)
        slope.append(MSD[-1]/len(MSD))
        WF3.plot(MSD, color='black', alpha=0.3)
        WF3.plot(lin,lin,linestyle='dotted',color='black')
        WF3.set_yticklabels([])
        #WF3.set_xticklabels([])
        #WF3.set_xlabel(r'$\tau$')
        WF3.set_xticks([200, 400])
        WF3.title.set_text(r'$WF_{3}$')

    elif item == "BT13B-3":
        for i in range(1,len(x)):
            if np.sqrt((x[i]-30)**2+(y[i]-30)**2) > 29:
                posx = 0
                posy = 0
                WF4.plot(MSD, color='black', alpha=0.3)
                slope.append(MSD[-1]/len(MSD))
                MSD = [0]
            else:
                MSD.append(np.sqrt(posx**2 + posy**2))
                posx += np.sqrt((x[i-1]-x[i])**2)
                posy += np.sqrt((y[i-1]-y[i])**2)
        slope.append(MSD[-1]/len(MSD))
        WF4.plot(MSD, color='black', alpha=0.3)
        WF4.plot(lin,lin,linestyle='dotted',color='black')
        WF4.set_yticklabels([])
        #WF4.set_xticklabels([])
        #WF4.set_xlabel(r'$\tau$')
        WF4.set_xticks([200, 400])
        WF4.title.set_text(r'$WF_{4}$')

    elif item == "BT03A-1":
        for i in range(1,len(x)):
            if np.sqrt((x[i]-30)**2+(y[i]-30)**2) > 29:
                posx = 0
                posy = 0
                RW1.plot(MSD, color='black', alpha=0.3)
                slope.append(MSD[-1]/len(MSD))
                MSD = [0]
            else:
                MSD.append(np.sqrt(posx**2 + posy**2))
                posx += np.sqrt((x[i-1]-x[i])**2)
                posy += np.sqrt((y[i-1]-y[i])**2)
        slope.append(MSD[-1]/len(MSD))
        RW1.plot(MSD, color='black', alpha=0.3)
        RW1.plot(lin,lin,linestyle='dotted',color='black')
        RW1.set_xlabel(r'$\tau$')
        RW1.set_ylabel(r'$\langle r^{2}(\tau) \rangle$')
        RW1.set_xticks([200, 400])
        RW1.title.set_text(r'$RW_{1}$')

    elif item == "BT06B-1":
        for i in range(1,len(x)):
            if np.sqrt((x[i]-30)**2+(y[i]-30)**2) > 29:
                posx = 0
                posy = 0
                RW2.plot(MSD, color='black', alpha=0.3)
                slope.append(MSD[-1]/len(MSD))
                MSD = [0]
            else:
                MSD.append(np.sqrt(posx**2 + posy**2))
                posx += np.sqrt((x[i-1]-x[i])**2)
                posy += np.sqrt((y[i-1]-y[i])**2)
        slope.append(MSD[-1]/len(MSD))
        RW2.plot(MSD, color='black', alpha=0.3)
        RW2.plot(lin,lin,linestyle='dotted',color='black')
        RW2.set_yticklabels([])
        #RW2.set_xticklabels([])
        RW2.set_xlabel(r'$\tau$')
        RW2.set_xticks([200, 400])
        RW2.title.set_text(r'$RW_{2}$')

    elif item == "BT13A-3":
        for i in range(1,len(x)):
            if np.sqrt((x[i]-30)**2+(y[i]-30)**2) > 29:
                posx = 0
                posy = 0
                RW3.plot(MSD, color='black', alpha=0.3)
                slope.append(MSD[-1]/len(MSD))
                MSD = [0]
            else:
                MSD.append(np.sqrt(posx**2 + posy**2))
                posx += np.sqrt((x[i-1]-x[i])**2)
                posy += np.sqrt((y[i-1]-y[i])**2)
        slope.append(MSD[-1]/len(MSD))
        RW3.plot(MSD, color='black', alpha=0.3)
        RW3.plot(lin,lin,linestyle='dotted',color='black')
        RW3.set_yticklabels([])
        #RW3.set_xticklabels([])
        RW3.set_xlabel(r'$\tau$')
        RW3.set_xticks([200, 400])
        RW3.title.set_text(r'$RW_{3}$')

    elif item == "BT12A-1":
        for i in range(1,len(x)):
            if np.sqrt((x[i]-30)**2+(y[i]-30)**2) > 29:
                posx = 0
                posy = 0
                RW4.plot(MSD, color='black', alpha=0.3)
                slope.append(MSD[-1]/len(MSD))
                MSD = [0]
            else:
                MSD.append(np.sqrt(posx**2 + posy**2))
                posx += np.sqrt((x[i-1]-x[i])**2)
                posy += np.sqrt((y[i-1]-y[i])**2)
        slope.append(MSD[-1]/len(MSD))
        RW4.plot(MSD, color='black', alpha=0.3)
        RW4.plot(lin,lin,linestyle='dotted',color='black')
        RW4.set_yticklabels([])
        #RW4.set_xticklabels([])
        RW4.set_xlabel(r'$\tau$')
        RW4.set_xticks([200, 400])
        RW4.title.set_text(r'$RW_{4}$')


fig, ((IB1, IB2, IB3, IB4), (GF1, GF2, GF3, GF4), (WF1, WF2, WF3, WF4), (RW1, RW2, RW3, RW4)) = plt.subplots(4, 4, figsize=(9,10.7))
fig.set_tight_layout(True)
for bee in Well_Behaved:
    MSD_bundle(bee)
plt.show()
"""------------------------------------------------------------------------------------------------------------------------------------------------"""
