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


fig, ((IB1, IB2, IB3, IB4), (GF1, GF2, GF3, GF4), (WF1, WF2, WF3, WF4), (RW1, RW2, RW3, RW4)) = plt.subplots(4, 4)

x,y = X_remove_NaN(IB_1),Y_remove_NaN(IB_1)
IB1.plot(ArenaX, ArenaY, color="black", linestyle="dashed")
IB1.plot(GradientX, GradientY, color="black", linestyle="dotted", dash_capstyle='round')
IB1.set_xticks([])
IB1.set_yticks([])
R = 0.0
G = 0.0
B = 1.0
time = len(x)
for i in range(time-1):
    IB1.plot((x[i],x[i+1]),(y[i],y[i+1]),color=(R,G,B))
    R+=(1/time)
    #G+=0.001
    B-=(1/time)

x,y = X_remove_NaN(IB_2),Y_remove_NaN(IB_2)
IB2.plot(ArenaX, ArenaY, color="black", linestyle="dashed")
IB2.plot(GradientX, GradientY, color="black", linestyle="dotted", dash_capstyle='round')
IB2.set_xticks([])
IB2.set_yticks([])
R = 0.0
G = 0.0
B = 1.0
time = len(x)
for i in range(time-1):
    IB2.plot((x[i],x[i+1]),(y[i],y[i+1]),color=(R,G,B))
    R+=(1/time)
    #G+=0.001
    B-=(1/time)

x,y = X_remove_NaN(IB_3),Y_remove_NaN(IB_3)
IB3.plot(ArenaX, ArenaY, color="black", linestyle="dashed")
IB3.plot(GradientX, GradientY, color="black", linestyle="dotted", dash_capstyle='round')
IB3.set_xticks([])
IB3.set_yticks([])
R = 0.0
G = 0.0
B = 1.0
time = len(x)
for i in range(time-1):
    IB3.plot((x[i],x[i+1]),(y[i],y[i+1]),color=(R,G,B))
    R+=(1/time)
    #G+=0.001
    B-=(1/time)

x,y = X_remove_NaN(IB_4),Y_remove_NaN(IB_4)
IB4.plot(ArenaX, ArenaY, color="black", linestyle="dashed")
IB4.plot(GradientX, GradientY, color="black", linestyle="dotted", dash_capstyle='round')
IB4.set_xticks([])
IB4.set_yticks([])
R = 0.0
G = 0.0
B = 1.0
time = len(x)
for i in range(time-1):
    IB4.plot((x[i],x[i+1]),(y[i],y[i+1]),color=(R,G,B))
    R+=(1/time)
    #G+=0.001
    B-=(1/time)

x,y = X_remove_NaN(GF_1),Y_remove_NaN(GF_1)
GF1.plot(ArenaX, ArenaY, color="black", linestyle="dashed")
GF1.plot(GradientX, GradientY, color="black", linestyle="dotted", dash_capstyle='round')
GF1.set_xticks([])
GF1.set_yticks([])
R = 0.0
G = 0.0
B = 1.0
time = len(x)
for i in range(time-1):
    GF1.plot((x[i],x[i+1]),(y[i],y[i+1]),color=(R,G,B))
    R+=(1/time)
    #G+=0.001
    B-=(1/time)

x,y = X_remove_NaN(GF_2),Y_remove_NaN(GF_2)
GF2.plot(ArenaX, ArenaY, color="black", linestyle="dashed")
GF2.plot(GradientX, GradientY, color="black", linestyle="dotted", dash_capstyle='round')
GF2.set_xticks([])
GF2.set_yticks([])
R = 0.0
G = 0.0
B = 1.0
time = len(x)
for i in range(time-1):
    GF2.plot((x[i],x[i+1]),(y[i],y[i+1]),color=(R,G,B))
    R+=(1/time)
    #G+=0.001
    B-=(1/time)

x,y = X_remove_NaN(GF_3),Y_remove_NaN(GF_3)
GF3.plot(ArenaX, ArenaY, color="black", linestyle="dashed")
GF3.plot(GradientX, GradientY, color="black", linestyle="dotted", dash_capstyle='round')
GF3.set_xticks([])
GF3.set_yticks([])
R = 0.0
G = 0.0
B = 1.0
time = len(x)
for i in range(time-1):
    GF3.plot((x[i],x[i+1]),(y[i],y[i+1]),color=(R,G,B))
    R+=(1/time)
    #G+=0.001
    B-=(1/time)

x,y = X_remove_NaN(GF_4),Y_remove_NaN(GF_4)
GF4.plot(ArenaX, ArenaY, color="black", linestyle="dashed")
GF4.plot(GradientX, GradientY, color="black", linestyle="dotted", dash_capstyle='round')
GF4.set_xticks([])
GF4.set_yticks([])
R = 0.0
G = 0.0
B = 1.0
time = len(x)
for i in range(time-1):
    GF4.plot((x[i],x[i+1]),(y[i],y[i+1]),color=(R,G,B))
    R+=(1/time)
    #G+=0.001
    B-=(1/time)

x,y = X_remove_NaN(WF_1),Y_remove_NaN(WF_1)
WF1.plot(ArenaX, ArenaY, color="black", linestyle="dashed")
WF1.plot(GradientX, GradientY, color="black", linestyle="dotted", dash_capstyle='round')
WF1.set_xticks([])
WF1.set_yticks([])
R = 0.0
G = 0.0
B = 1.0
time = len(x)
for i in range(time-1):
    WF1.plot((x[i],x[i+1]),(y[i],y[i+1]),color=(R,G,B))
    R+=(1/time)
    #G+=0.001
    B-=(1/time)

x,y = X_remove_NaN(WF_2),Y_remove_NaN(WF_2)
WF2.plot(ArenaX, ArenaY, color="black", linestyle="dashed")
WF2.plot(GradientX, GradientY, color="black", linestyle="dotted", dash_capstyle='round')
WF2.set_xticks([])
WF2.set_yticks([])
R = 0.0
G = 0.0
B = 1.0
time = len(x)
for i in range(time-1):
    WF2.plot((x[i],x[i+1]),(y[i],y[i+1]),color=(R,G,B))
    R+=(1/time)
    #G+=0.001
    B-=(1/time)

x,y = X_remove_NaN(WF_3),Y_remove_NaN(WF_3)
WF3.plot(ArenaX, ArenaY, color="black", linestyle="dashed")
WF3.plot(GradientX, GradientY, color="black", linestyle="dotted", dash_capstyle='round')
WF3.set_xticks([])
WF3.set_yticks([])
R = 0.0
G = 0.0
B = 1.0
time = len(x)
for i in range(time-1):
    WF3.plot((x[i],x[i+1]),(y[i],y[i+1]),color=(R,G,B))
    R+=(1/time)
    #G+=0.001
    B-=(1/time)

x,y = X_remove_NaN(WF_4),Y_remove_NaN(WF_4)
WF4.plot(ArenaX, ArenaY, color="black", linestyle="dashed")
WF4.plot(GradientX, GradientY, color="black", linestyle="dotted", dash_capstyle='round')
WF4.set_xticks([])
WF4.set_yticks([])
R = 0.0
G = 0.0
B = 1.0
time = len(x)
for i in range(time-1):
    WF4.plot((x[i],x[i+1]),(y[i],y[i+1]),color=(R,G,B))
    R+=(1/time)
    #G+=0.001
    B-=(1/time)

x,y = X_remove_NaN(RW_1),Y_remove_NaN(RW_1)
RW1.plot(ArenaX, ArenaY, color="black", linestyle="dashed")
RW1.plot(GradientX, GradientY, color="black", linestyle="dotted", dash_capstyle='round')
RW1.set_xticks([])
RW1.set_yticks([])
R = 0.0
G = 0.0
B = 1.0
time = len(x)
for i in range(time-1):
    RW1.plot((x[i],x[i+1]),(y[i],y[i+1]),color=(R,G,B))
    R+=(1/time)
    #G+=0.001
    B-=(1/time)

x,y = X_remove_NaN(RW_2),Y_remove_NaN(RW_2)
RW2.plot(ArenaX, ArenaY, color="black", linestyle="dashed")
RW2.plot(GradientX, GradientY, color="black", linestyle="dotted", dash_capstyle='round')
RW2.set_xticks([])
RW2.set_yticks([])
R = 0.0
G = 0.0
B = 1.0
time = len(x)
for i in range(time-1):
    RW2.plot((x[i],x[i+1]),(y[i],y[i+1]),color=(R,G,B))
    R+=(1/time)
    #G+=0.001
    B-=(1/time)

x,y = X_remove_NaN(RW_3),Y_remove_NaN(RW_3)
RW3.plot(ArenaX, ArenaY, color="black", linestyle="dashed")
RW3.plot(GradientX, GradientY, color="black", linestyle="dotted", dash_capstyle='round')
RW3.set_xticks([])
RW3.set_yticks([])
R = 0.0
G = 0.0
B = 1.0
time = len(x)
for i in range(time-1):
    RW3.plot((x[i],x[i+1]),(y[i],y[i+1]),color=(R,G,B))
    R+=(1/time)
    #G+=0.001
    B-=(1/time)

x,y = X_remove_NaN(RW_4),Y_remove_NaN(RW_4)
RW4.plot(ArenaX, ArenaY, color="black", linestyle="dashed")
RW4.plot(GradientX, GradientY, color="black", linestyle="dotted", dash_capstyle='round')
RW4.set_xticks([])
RW4.set_yticks([])
R = 0.0
G = 0.0
B = 1.0
time = len(x)
for i in range(time-1):
    RW4.plot((x[i],x[i+1]),(y[i],y[i+1]),color=(R,G,B))
    R+=(1/time)
    #G+=0.001
    B-=(1/time)
plt.show()
