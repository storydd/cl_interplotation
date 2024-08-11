import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import pyplot
from mpl_toolkits.mplot3d import axes3d

with open('loss.txt', 'r') as file:
    lines = file.readlines()
print(len(lines))
loss=[]
val_loss=[]
for i in range(1000):
    words=lines[i*4+3].split(' ')
    print(words)
    loss.append(float(words[7]))
    val_loss.append(float(words[13]))
print(loss)
print(val_loss)
# 绘制损失图
# 设置字体样式为Arial
matplotlib.rcParams['font.family'] = 'Times New Roman'
pyplot.plot(range(1000),loss,ls='-',label='train_Loss',c=(0.5, 0.8, 0.9))
pyplot.plot(range(1000),val_loss,ls='-',label='val_Loss',c=(0.9, 0.5, 0.5))
pyplot.xlabel('Epoch')
pyplot.ylabel('Loss')
# 添加图例
pyplot.legend()
# pyplot.show()
pyplot.savefig('Fig5.png',dpi=300)