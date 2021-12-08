import numpy as np
import matplotlib.pyplot as plt
import math

point1 = [0, 1]
point2 = [1, 1]
point3 = [1, 0]
point4 = [2, 0]
point5 = [2.5, 0]
point6 = [2.5, 1]
point7 = [3, 1]
point8 = [3, 0]
point9 = [4.5, 0]
point10 = [4.5, 1]
point11= [5.3, 1]
point12 = [5.3, 0]
point13 = [6, 0]
point14 = [6, 1]
point15 = [6.5, 1]
point16 = [6.5, 0]
point17 = [6.8, 0]
point18= [6.8, 1]
point19 = [8.3, 1]
point20 = [8.3, 0]
point21 = [9, 0]
point22 = [9, 1]
point23 = [9.6, 1]
point24 = [9.6, 0]
point25 = [10, 0]
point26 = [11, 0]
point27 = [11, 1]
x = np.linspace(0,12,12)
y = [0.5 for i in range(12)]

x_values = [point1[0], point2[0], point3[0], point4[0], point5[0], point6[0], point7[0], point8[0], point9[0], point10[0], point11[0], point12[0], point13[0], point14[0], point15[0], point16[0], point17[0], point18[0], point19[0], point20[0], point21[0], point22[0], point23[0], point24[0], point25[0], point26[0], point27[0]]
y_values = [point1[1], point2[1], point3[1], point4[1], point5[1], point6[1], point7[1], point8[1], point9[1], point10[1], point11[1], point12[1], point13[1], point14[1], point15[1], point16[1], point17[1], point18[1], point19[1], point20[1], point21[1], point22[1], point23[1], point24[1], point25[1], point26[1], point27[1]]




fig = plt.figure(figsize=(8,3))
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8]) # main axes

ax.plot(x_values, y_values,color='black')
ax.plot(x,y,linestyle='dotted',color='black')
ax.set_xticks([])
ax.set_yticks([0,1])
ax.set_yticklabels([r'$-v_{0}/2$', r'$v_{0}/2$'])
plt.show()
