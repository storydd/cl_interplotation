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



dataset=np.zeros([75, 4, 7])
labelset=np.zeros([75, 4, 4])
for i in range(75):
    dataset[i, :, :] = data[0 + i * 4: 4 + i * 4,:]
for i in range(75):
    labelset[i, :, :] = label[0 + i * 4: 4 + i * 4,:]


print(dataset.shape,labelset.shape)




dataset = np.transpose(dataset, (1, 0, 2))
labelset = np.transpose(labelset, (1, 0, 2))
print("数据形状:", dataset.shape)
print("标签形状:", labelset.shape)
# # # #3DCNN的输入为（4, 4, 3, 7）,4：每次卷积4个深度，4，3：每层的点位数，7：特征数
# 打印合并后的数组形状
#将数据合并为3DCNN所需的格式
data=np.zeros([128,3,4,3,7])
label=np.zeros([128,3,4,3,4])


for i in range(64):
    data[i,:,:,:,:]=dataset[0:3,i:i+12,:].reshape([3,4,3,7])
    label[i, :, :, :] = labelset[0:3,i:i+12,:].reshape([3,4,3,4])
for i in range(64,128):
    data[i,:,:,:,:]=dataset[1:4,i-64:i-64+12,:].reshape([3,4,3,7])
    label[i, :, :, :] = labelset[1:4,i-64:i-64+12,:].reshape([3,4,3,4])
#
print("数据形状:", data.shape)
print("标签形状:", label.shape)







X_train, X_val, y_train, y_val = model_selection.train_test_split(data, label, test_size = 0.36, random_state = 5)
X_val, X_test, y_val, y_test = model_selection.train_test_split(X_val, y_val, test_size = 0.5, random_state = 12)


X_train=X_train.reshape([2916,7])
X_val=X_val.reshape([828,7])
X_test=X_test.reshape([864,7])
y_train=y_train.reshape([2916,4])
y_val=y_val.reshape([828,4])
y_test=y_test.reshape([864,4])
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


from sklearn.ensemble import RandomForestRegressor

# 创建随机森林回归模型
rf = RandomForestRegressor(n_estimators=100, max_depth=10)


# 使用训练数据拟合模型
y_Cdrf=rf.fit(X_train, y_train[:,0])
y_Pbrf=rf.fit(X_train, y_train[:,1])
y_Curf=rf.fit(X_train, y_train[:,2])
y_Nirf=rf.fit(X_train, y_train[:,3])
# 使用模型进行预测
y_Cdpred = y_Cdrf.predict(X_test)
y_Pbpred = y_Pbrf.predict(X_test)
y_Cupred = y_Curf.predict(X_test)
y_Nipred = y_Nirf.predict(X_test)


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