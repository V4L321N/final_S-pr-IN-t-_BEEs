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


print(len(grad_TEN_delta))
print(len(grad_EIGHT_delta))
print(len(grad_SIX_delta))
print(len(grad_FOUR_delta))
print(len(grad_TWO_delta))
print(len(grad_NO_delta))

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

print(grad_TWO_delta)
