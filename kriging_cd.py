import numpy as np
import pandas as pd
from pykrige import OrdinaryKriging3D
from sklearn import model_selection
import keras.backend as K
from keras.callbacks import LearningRateScheduler
from sklearn.metrics import r2_score
from tensorflow import keras
from tensorflow.keras.models import Sequential
from matplotlib import pyplot
# 数据归一化通常指的是将特征缩放到一个指定的最小和最大值（通常是0到1）之间。这可以通过MinMaxScaler来完成。
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.layers import Conv3DTranspose

from keras.models import Sequential
from keras.layers import Conv3D, Dense, Flatten, Activation, GlobalAveragePooling3D, Layer, Conv3DTranspose, Layer, \
    GlobalAveragePooling3D, Reshape, Dense, Multiply, Dropout, BatchNormalization

from matplotlib import pyplot
from sklearn import model_selection


#读取数据
data=pd.read_excel(open('data/data.xlsx', 'rb'), usecols=[1, 2, 3, 4, 5, 6, 7])
label=pd.read_excel(open('data/data.xlsx', 'rb'), usecols=[9,10,11,12])


data=np.array(data)
label=np.array(label)

print(data.shape)
print(label.shape)


X_train, X_val, y_train, y_val = model_selection.train_test_split(data, label, test_size = 0.36, random_state = 5)
X_val, X_test, y_val, y_test = model_selection.train_test_split(X_val, y_val, test_size = 0.5, random_state = 12)
#处理数据
# 初始化MinMaxScaler
scaler_x = MinMaxScaler()
# 拟合数据并进行转换
X_train = scaler_x.fit_transform(X_train)
X_val = scaler_x.transform(X_val)
X_test = scaler_x.transform(X_test)
# 初始化MinMaxScaler
scaler_y = MinMaxScaler()
# 拟合数据并进行转换
y_train = scaler_y.fit_transform(y_train)
y_val = scaler_y.transform(y_val)
y_test = scaler_y.transform(y_test)

print("训练集数据形状:", X_train.shape)
print("训练集标签形状:", X_val.shape)
print("验证集数据形状:", X_test.shape)
print("验证集标签形状:", y_train.shape)
print("测试集数据形状:", y_val.shape)
print("测试集标签形状:", y_test.shape)
# ok3d = OrdinaryKriging3D(train_data[:, 0], train_data[:, 1], train_data[:, 2], train_data[:, 3], variogram_model="linear",verbose=1)
ok3d_Cd = OrdinaryKriging3D(X_train[:, 0], X_train[:, 1], X_train[:, 2], y_train[:,0], variogram_model="gaussian")
ok3d_Pb = OrdinaryKriging3D(X_train[:, 0], X_train[:, 1], X_train[:, 2], y_train[:,1], variogram_model="gaussian")
ok3d_Cu = OrdinaryKriging3D(X_train[:, 0], X_train[:, 1], X_train[:, 2], y_train[:,2], variogram_model="gaussian")
ok3d_Ni = OrdinaryKriging3D(X_train[:, 0], X_train[:, 1], X_train[:, 2], y_train[:,3], variogram_model="gaussian")
# variogram_model（str，可选） - 指定要使用的变异函数模型; 可能是以下之一：线性(linear)，幂(power)，高斯(gaussian)，球形(spherical)，指数(exponential)，孔效应(hole-effect)。 默认是线性变异函数模型。
#
#
#
# #预测
y_Cdpred, Cd3d = ok3d_Cd.execute("points", X_test[:, 0], X_test[:, 1], X_test[:, 2])
y_Pbpred, Pb3d = ok3d_Pb.execute("points", X_test[:, 0], X_test[:, 1], X_test[:, 2])
y_Cupred, Cu3d = ok3d_Cu.execute("points", X_test[:, 0], X_test[:, 1], X_test[:, 2])
y_Nipred, Ni3d = ok3d_Ni.execute("points", X_test[:, 0], X_test[:, 1], X_test[:, 2])
#linear、power、gaussian、spherical、exponential、hole-effect

#
print('真实结果形状',y_test.shape)
print('预测结果形状',y_Cdpred.shape)
print('预测结果形状',y_Pbpred.shape)
print('预测结果形状',y_Cupred.shape)
print('预测结果形状',y_Nipred.shape)



# print(sum(Y_train[:, 0])/len(Y_train[:, 0]))
# 计算MSE
rmean_squared_error = np.sqrt(np.mean((y_test[:,0] - y_Cdpred) ** 2))

# 计算MAE
mean_absolute_error = np.mean(np.abs(y_test[:,0] - y_Cdpred))
R2 = r2_score(y_test[:,0],y_Cdpred)

print('RMSE:', rmean_squared_error)
print('MAE:', mean_absolute_error)

print('R2:',R2)

######################
# print(sum(Y_train[:, 0])/len(Y_train[:, 0]))
# 计算MSE
rmean_squared_error = np.sqrt(np.mean((y_test[:,1] - y_Pbpred) ** 2))

# 计算MAE
mean_absolute_error = np.mean(np.abs(y_test[:,1] - y_Pbpred))
R2 = r2_score(y_test[:,1],y_Pbpred)

print('RMSE:', rmean_squared_error)
print('MAE:', mean_absolute_error)

print('R2:',R2)
########################
# print(sum(Y_train[:, 0])/len(Y_train[:, 0]))
# 计算MSE
rmean_squared_error = np.mean(((y_test[:,2] - y_Cupred) ** 2))

# 计算MAE
mean_absolute_error = np.mean(np.abs(y_test[:,2] - y_Cupred))
R2 = r2_score(y_test[:,2],y_Cupred)

print('RMSE:', rmean_squared_error)
print('MAE:', mean_absolute_error)

print('R2:',R2)
#########################
# print(sum(Y_train[:, 0])/len(Y_train[:, 0]))
# 计算MSE
rmean_squared_error = np.sqrt(np.mean((y_test[:,3] - y_Nipred) ** 2))

# 计算MAE
mean_absolute_error = np.mean(np.abs(y_test[:,3] - y_Nipred))
R2 = r2_score(y_test[:,3],y_Nipred)

print('RMSE:', rmean_squared_error)
print('MAE:', mean_absolute_error)

print('R2:',R2)