import matplotlib.pyplot as plt
import numpy as np

# 生成数据
true_values = np.array([0, 1.1, 2.0, 2.5, 3.3, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5,8])
predicted_values = 1 * true_values + 0.5 + np.random.randn(12)  # 生成对应的 y 值，其中 y=3x+2 是线性关系，加上一些噪声
fig, ax = plt.subplots()
# 画出散点图
plt.scatter(true_values, predicted_values)
#设置字体
plt.rcParams['font.family'] = 'Times New Roman'
plt.rc('mathtext', default='regular')

plt.plot(true_values, predicted_values, 'o', c=(0.5, 0.8, 0.5), markersize=6)
# 拟合直线
slope, intercept = np.polyfit(true_values, predicted_values, 1)
plt.plot(true_values, 1.3 * true_values - 1.0, ls='--',color='green')
plt.plot(true_values, true_values, color='blue')
# 设置x轴的范围
plt.xlim(0, 8)  # x轴的范围设置为0到5
plt.ylim(0, 8)
# 显示图形
plt.xticks(fontproperties='Times New Roman', size=15)

plt.yticks(fontproperties='Times New Roman', size=15)
plt.show()