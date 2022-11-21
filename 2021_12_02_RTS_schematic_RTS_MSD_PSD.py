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
x = np.linspace(-1,12,12)
y = [0.5 for i in range(12)]
x_values = [point1[0], point2[0], point3[0], point4[0], point5[0], point6[0], point7[0], point8[0], point9[0], point10[0], point11[0], point12[0], point13[0], point14[0], point15[0], point16[0], point17[0], point18[0], point19[0], point20[0], point21[0], point22[0], point23[0], point24[0], point25[0], point26[0], point27[0]]
y_values = [point1[1], point2[1], point3[1], point4[1], point5[1], point6[1], point7[1], point8[1], point9[1], point10[1], point11[1], point12[1], point13[1], point14[1], point15[1], point16[1], point17[1], point18[1], point19[1], point20[1], point21[1], point22[1], point23[1], point24[1], point25[1], point26[1], point27[1]]

fig = plt.figure(figsize=(8,3))
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8]) # main axes
# ax.arrow(point3[0], point3[1]+0.1, 1.37, 0,
#           head_width = 0.06,
#           width = 0.03,
#           ec ='blue')
ax.annotate(r'$t_{s}$', (point3[0]+0.6, point3[1]+0.15))
ax.annotate(s='', xy=(0.95, 0.1), xytext=(2.55,0.1), arrowprops=dict(arrowstyle='<->', color='blue', linewidth=2))

# ax.arrow(point17[0], point17[1]+0.9, 1.37, 0,
#           head_width = 0.06,
#           width = 0.03,
#           color = 'firebrick',
#           ec ='red')

ax.annotate(r'$t_{w}$', (point17[0]+0.6, point17[1]+0.8))
ax.annotate(s='', xy=(6.75, 0.9), xytext=(8.35,0.9), arrowprops=dict(arrowstyle='<->', color='red', linewidth=2))
ax.plot(x_values, y_values,color='black')
ax.plot(x,y,linestyle='dotted',color='black')
ax.set_xticks([])
ax.set_yticks([0,1])
ax.set_yticklabels([r'$-v_{0}/2$', r'$v_{0}/2$'])
ax.set_xlim(0,10)
ax.set_xlabel('time')
ax.set_ylabel('RTS')
plt.show()

# x1 = np.linspace(-2.5,4)
#
# plt.figure(figsize=(10,5))
# plt.xticks([])
# plt.yticks([])
# plt.xlabel(r'$\tau$')
# plt.ylabel(r'$MSD$')
# plt.xlim(-2.2,3.7)
# plt.plot(x1, np.tanh(x1)+5, color='black')
# plt.axvline(x=-2, linestyle='dotted', color='black')
# plt.annotate(r'$C_{I}\tau^{\alpha}, \alpha > 1$', (-1.6, 5.5))
# plt.annotate('super-diffusive', (-1.7, 3.95))
# plt.axvline(x=-0.5, linestyle='dotted', color='black')
# plt.annotate(r'$C_{II}\tau^{\alpha}, \alpha = 1$', (-0.25, 5.5))
# plt.annotate('normal-diffusive', (-0.42, 3.95))
# plt.axvline(x=0.5, linestyle='dotted', color='black')
# plt.annotate(r'$C_{III}\tau^{\alpha}, \alpha < 1$', (1, 5.5))
# plt.annotate('sub-diffusive', (0.92, 3.95))
# plt.axvline(x=2, linestyle='dotted', color='black')
# plt.annotate(r'$C_{IV}\tau^{\alpha}, \alpha = 0$', (2.4, 5.5))
# plt.annotate('saturation', (2.5, 3.95))
# plt.axvline(x=3.5, linestyle='dotted', color='black')
# plt.show()

# def S(omega):
#     D = 30
#     a = 300
#     return (2 * D) / (a**2 + omega**2)
#
# x = np.linspace(0,100)
# otherx = np.linspace(0,120)
# plt.figure(figsize=(10,5))
# plt.xlim(10,120)
# plt.ylim(6*10**(-4), 6.8*10**(-4))
# plt.plot(x, S(x))
# plt.plot(otherx, 0.000666214*otherx/otherx)
# plt.xscale('log')
# plt.yscale('log')
# plt.show()
#
# x = [0, 10, 15, 17.5]
# y = [10, 10, 5, 0]
# plt.figure(figsize=(8,5))
# plt.xlim(0,18.081988)
# plt.ylim(0,11.5)
# plt.xticks([])
# plt.yticks([])
# plt.xlabel(r'$ln(\omega)$')
# plt.ylabel(r'$ln(S_{xx})$')
# plt.plot(x, y, color='black')
# plt.axvline(x=10, linestyle='dotted', color='black')
# plt.axvline(x=15, linestyle='dotted', color='black')
# plt.annotate(r'$\frac{1}{\omega^{0}}$', (5,10.6), size=15)
# plt.annotate(r'$\frac{1}{\omega^{1}}$', (12.4,10.6), size=15)
# plt.annotate(r'$\frac{1}{\omega^{2}}$', (16.4,10.6), size=15)
# plt.show()
