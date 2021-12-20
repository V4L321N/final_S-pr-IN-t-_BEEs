import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.patches as patches

ArenaX = pd.DataFrame(data=pd.read_csv("data_bee_types/Arena.csv"), columns=["ArenaX"])
ArenaY = pd.DataFrame(data=pd.read_csv("data_bee_types/Arena.csv"), columns=["ArenaY"])
GradientX = pd.DataFrame(data=pd.read_csv("data_bee_types/Arena.csv"), columns=["GradX"])
GradientY = pd.DataFrame(data=pd.read_csv("data_bee_types/Arena.csv"), columns=["GradY"])

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


fig, ((IB1, IB2, IB3, IB4), (GF1, GF2, GF3, GF4), (WF1, WF2, WF3, WF4), (RW1, RW2, RW3, RW4)) = plt.subplots(4, 4, figsize=(10,10))
itemize = [IB1, IB2, IB3, IB4, GF1, GF2, GF3, GF4, WF1, WF2, WF3, WF4, RW1, RW2, RW3, RW4]
type = [r'$IB_1$', r'$IB_2$', r'$IB_3$', r'$IB_4$', r'$GF_1$', r'$GF_2$', r'$GF_3$', r'$GF_4$', r'$WF_1$', r'$WF_2$', r'$WF_3$', r'$WF_4$', r'$RW_1$', r'$RW_2$', r'$RW_3$', r'$RW_4$']
n=0
fig.set_tight_layout(True)
for item in itemize:
    x,y = X_remove_NaN(Well_Behaved[n]),Y_remove_NaN(Well_Behaved[n])
    item.plot(ArenaX, ArenaY, color="black", linestyle="dashed")
    item.plot(GradientX, GradientY, color="black", linestyle="dotted", dash_capstyle='round')
    item.annotate(type[n], xy=(0.8,0.9),xycoords='axes fraction', fontsize=12)
    item.set_xticks([])
    item.set_yticks([])
    R = 0.0
    G = 0.0
    B = 1.0
    time = len(x)
    for i in range(time-1):
        item.plot((x[i],x[i+1]),(y[i],y[i+1]),color=(R,G,B))
        R+=(1/time)
        #G+=0.001
        B-=(1/time)
    n += 1
plt.show()
