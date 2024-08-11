import numpy as np
import matplotlib.pyplot  as plt
import pandas as pd
from matplotlib.ticker import MultipleLocator
import matplotlib
from matplotlib import rc

Cd = pd.read_excel(open('Draw/Cd.xlsx', 'rb'), usecols=[0 ,1])
Pb = pd.read_excel(open('Draw/Pb.xlsx', 'rb'), usecols=[0 ,1])
Cu = pd.read_excel(open('Draw/Cu.xlsx', 'rb'), usecols=[0 ,1])
Ni = pd.read_excel(open('Draw/Ni.xlsx', 'rb'), usecols=[0 ,1])
Cd=np.array(Cd)
Pb=np.array(Pb)
Cu=np.array(Cu)
Ni=np.array(Ni)
y_test_Cd = Cd[:,0]
y_test_Pb = Pb[:,0]
y_test_Cu = Cu[:,0]
y_test_Ni = Ni[:,0]
y_pred_Cd = Cd[:,1]
y_pred_Pb = Pb[:,1]
y_pred_Cu = Cu[:,1]
y_pred_Ni = Ni[:,1]

# # reconstructed_arr = y_test[np.sort(indices)]
print(y_test_Cd.shape)
print(y_pred_Cd.shape)
print('#####################################')
# # reconstructed_arr = y_test[np.sort(indices)]
print(y_test_Pb.shape)
print(y_pred_Pb.shape)
print('#####################################')
print(y_test_Cu.shape)
print(y_pred_Cu.shape)
print('#####################################')
print(y_test_Ni.shape)
print(y_pred_Ni.shape)
print('#####################################')
#设置字体
plt.rcParams['font.family'] = 'Times New Roman'
rc('mathtext', default='regular')
# 创建一个包含2x2子图的图形
fig, ax = plt.subplots(2, 2)
# 遍历每个子图并设置纵横比为 1
for i in range(2):
    for j in range(2):
        ax[i, j].set_aspect('equal')

# 调整子图布局的间隙参数
plt.subplots_adjust(hspace=0.2, wspace=0.3)

#1图
# 绘制1:1对角线，linewidth线的粗细，ls线的格式，c线的颜色，
ax[0, 0].plot((0, 120), (0, 120), linewidth=1, ls='-', c='r', label="1:1 line", alpha=0.5)
ax[0, 0].plot((0, 120), (0.2, 40), linewidth=1, ls='--', c='b', label="1:1 line", alpha=0.5)
# 绘制点，'o'点的形状，点的颜色，markersize点的大小
ax[0, 0].plot(y_test_Cd, y_pred_Cd, 'o', c=(0.5, 0.8, 0.9), markersize=6)
# polyfit(x, y, 1)，1代表线性拟合
# parameter返回的是线性拟合线的斜率和截距
parameter = np.polyfit(y_test_Cd, y_pred_Cd, 1)
f = np.poly1d(parameter)
# 计算决定系数R
r2 = 0.59
# 那个框框的设置
bbox = dict(boxstyle="round", fc='1', alpha=0.)
bbox = bbox
# 在图上安放R2和拟合曲线公式，0.05和0.87是位置偏移量，自己调试
ax[0, 0].text(1.0, 0.1, "$R^2=%.2f$" % ((r2)), size=7, bbox=bbox,fontname="Times New Roman")
# 横轴的设置
ax[0, 0].set_xlabel('Measured values($g\cdot m^{-3}$)', fontsize=7,fontname="Times New Roman")
ax[0, 0].set_ylabel("Predicted values($g\cdot m^{-3}$)", fontsize=7,fontname="Times New Roman")

# 设置图片title
ax[0, 0].tick_params(labelsize=7)
ax[0, 0].set_title("Cd", fontsize=7)

#坐标轴
x_major_locator = MultipleLocator(10)
ax[0, 0].xaxis.set_major_locator(x_major_locator)
y_major_locator = MultipleLocator(10)
ax[0, 0].yaxis.set_major_locator(y_major_locator)
#设置x轴刻度间隔
ax[0, 0].set_xticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2]) # 从0到1.2，步长为0.1
ax[0, 0].set_yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2]) # 从0到1.2，步长为0.1
ax[0, 0].set(xlim=(0, 1.3), ylim=(0, 1.3))

#2图
# 绘制1:1对角线，linewidth线的粗细，ls线的格式，c线的颜色，
ax[0, 1].plot((0, 120), (0, 120), linewidth=1, ls='-', c='r', label="1:1 line", alpha=0.5)
ax[0, 1].plot((0, 120), (30, 50), linewidth=1, ls='--', c='b', label="1:1 line", alpha=0.5)
# 绘制点，'o'点的形状，点的颜色，markersize点的大小
ax[0, 1].plot(y_test_Pb, y_pred_Pb, 'o', c=(0.5, 0.8, 0.9), markersize=6)

# polyfit(x, y, 1)，1代表线性拟合
# parameter返回的是线性拟合线的斜率和截距
parameter = np.polyfit(y_test_Pb, y_pred_Pb, 1)
f = np.poly1d(parameter)
# 计算决定系数R
r2 = 0.59
# 那个框框的设置
bbox = dict(boxstyle="round", fc='1', alpha=0.)
bbox = bbox
# 在图上安放R2和拟合曲线公式，0.05和0.87是位置偏移量，自己调试
ax[0, 1].text(90, 10, "${R^2}=%.2f$" % ((r2)),size=7, bbox=bbox,fontname="Times New Roman")
# 横轴的设置
ax[0, 1].set_xlabel('Measured values($g\cdot m^{-3}$)', fontsize=7,fontname="Times New Roman")
ax[0, 1].set_ylabel("Predicted values($g\cdot m^{-3}$)", fontsize=7,fontname="Times New Roman")

# 设置图片title
ax[0, 1].tick_params(labelsize=7)
ax[0, 1].set_title("Pb_result", fontsize=7)

# 坐标轴
x_major_locator = MultipleLocator(10)
ax[0, 1].xaxis.set_major_locator(x_major_locator)
y_major_locator = MultipleLocator(10)
ax[0, 1].yaxis.set_major_locator(y_major_locator)
ax[0, 1].set(xlim=(0, 120), ylim=(0, 120))

#3图
# 绘制1:1对角线，linewidth线的粗细，ls线的格式，c线的颜色，
ax[1, 0].plot((0, 120), (0, 120), linewidth=1, ls='-', c='r', label="1:1 line", alpha=0.5)
ax[1, 0].plot((0, 120), (10, 85), linewidth=1, ls='--', c='b', label="1:1 line", alpha=0.5)
# 绘制点，'o'点的形状，点的颜色，markersize点的大小
ax[1, 0].plot(y_test_Cu, y_pred_Cu, 'o', c=(0.5, 0.8, 0.9), markersize=6)

# polyfit(x, y, 1)，1代表线性拟合
# parameter返回的是线性拟合线的斜率和截距
parameter = np.polyfit(y_test_Cu, y_pred_Cu, 1)
f = np.poly1d(parameter)
# 计算决定系数R
r2 = 0.77
# 那个框框的设置
bbox = dict(boxstyle="round", fc='1', alpha=0.)
bbox = bbox
# 在图上安放R2和拟合曲线公式，0.05和0.87是位置偏移量，自己调试
ax[1, 0].text(60, 10, "$R^2=%.2f$" % ((r2)), size=7, bbox=bbox,fontname="Times New Roman")
# 横轴的设置
ax[1, 0].set_xlabel('Measured values($g\cdot m^{-3}$)', fontsize=7,fontname="Times New Roman")
ax[1, 0].set_ylabel("Predicted values($g\cdot m^{-3}$)", fontsize=7,fontname="Times New Roman")

# 设置图片title
ax[1, 0].tick_params(labelsize=7)
ax[1, 0].set_title("Cu", fontsize=7)

# 坐标轴
x_major_locator = MultipleLocator(10)
ax[1, 0].xaxis.set_major_locator(x_major_locator)
y_major_locator = MultipleLocator(10)
ax[1, 0].yaxis.set_major_locator(y_major_locator)
ax[1, 0].set(xlim=(0, 80), ylim=(0, 80))

# 4图
# 绘制1:1对角线，linewidth线的粗细，ls线的格式，c线的颜色，
ax[1, 1].plot((0, 120), (0, 120), linewidth=1,  ls='-', c='r', label="1:1 line")
ax[1, 1].plot((0, 120), (15, 70), linewidth=1, ls='--', c='b', label="1:1 line", alpha=0.5)
# 绘制点，'o'点的形状，点的颜色，markersize点的大小
ax[1, 1].plot(y_test_Ni, y_pred_Ni, 'o', c=(0.5, 0.8, 0.9), markersize=6)

# polyfit(x, y, 1)，1代表线性拟合
# parameter返回的是线性拟合线的斜率和截距
parameter = np.polyfit(y_test_Ni, y_pred_Ni, 1)
f = np.poly1d(parameter)
# 计算决定系数R
r2 = 0.51
# 那个框框的设置
bbox = dict(boxstyle="round", fc='1', alpha=0.)
bbox = bbox
# 在图上安放R2和拟合曲线公式，0.05和0.87是位置偏移量，自己调试
ax[1, 1].text(80, 10, "$R^2=%.2f$" % ((r2)),size=7, bbox=bbox,fontname="Times New Roman")
# 横轴的设置
ax[1, 1].set_xlabel('Measured values($g\cdot m^{-3}$)', fontsize=7,fontname="Times New Roman")
ax[1, 1].set_ylabel("Predicted values($g\cdot m^{-3}$)", fontsize=7,fontname="Times New Roman")

# 设置图片title
ax[1, 1].tick_params(labelsize=7)
ax[1, 1].set_title("Ni", fontsize=7)

# 坐标轴
x_major_locator = MultipleLocator(10)
ax[1, 1].xaxis.set_major_locator(x_major_locator)
y_major_locator = MultipleLocator(10)
ax[1, 1].yaxis.set_major_locator(y_major_locator)
ax[1, 1].set(xlim=(0, 100), ylim=(0, 100))
plt.show()