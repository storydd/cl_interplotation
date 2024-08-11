import keras.models
import matplotlib
import numpy as np
import pandas as pd
from keras import Sequential
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Conv3D, Dropout, Conv3DTranspose
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from sklearn import model_selection
from tensorflow.python.layers.base import Layer
import tensorflow as tf

#读取数据
data=pd.read_excel(open('data/data.xlsx', 'rb'), usecols=[1, 2, 3, 4, 5, 6, 7])
label=pd.read_excel(open('data/data.xlsx', 'rb'), usecols=[9])

data=np.array(data)
label=np.array(label)

dataset=np.zeros([75, 4, 7])
labelset=np.zeros([75, 4, 1])
for i in range(75):
    dataset[i, :, :] = data[0 + i * 4: 4 + i * 4,:]
for i in range(75):
    labelset[i, :, :] = label[0 + i * 4: 4 + i * 4,:]

dataset = np.transpose(dataset, (1, 0, 2))
labelset = np.transpose(labelset, (1, 0, 2))

data=np.zeros([128,3,4,3,7])
label=np.zeros([128,3,4,3,1])

for i in range(64):
    data[i,:,:,:,:]=dataset[0:3,i:i+12,:].reshape([3,4,3,7])
    label[i, :, :, :] = labelset[0:3,i:i+12,:].reshape([3,4,3,1])
for i in range(64,128):
    data[i,:,:,:,:]=dataset[1:4,i-64:i-64+12,:].reshape([3,4,3,7])
    label[i, :, :, :] = labelset[1:4,i-64:i-64+12,:].reshape([3,4,3,1])


X_train, X_val, y_train, y_val = model_selection.train_test_split(data, label, test_size = 0.36, random_state = 1234)
X_val, X_test, y_val, y_test = model_selection.train_test_split(X_val, y_val, test_size = 0.5, random_state = 1234)

X_train=X_train.reshape([2916,7])
X_val=X_val.reshape([828,7])
X_test=X_test.reshape([864,7])
y_train=y_train.reshape([2916,1])
y_val=y_val.reshape([828,1])
y_test=y_test.reshape([864,1])
#处理数据
# 初始化MinMaxScaler
scaler_x = MinMaxScaler()
# 拟合数据并进行转换
scaler_x.fit(X_train)
X_train = scaler_x.transform(X_train)
X_val = scaler_x.transform(X_val)
X_test = scaler_x.transform(X_test)
# 初始化MinMaxScaler
scaler_y = MinMaxScaler()
# 拟合数据并进行转换
scaler_y.fit(y_train)
y_train = scaler_y.transform(y_train)
y_val = scaler_y.transform(y_val)
y_test = scaler_y.transform(y_test)
X_train=X_train.reshape([81, 3, 4, 3,7])
X_val=X_val.reshape([23, 3, 4, 3, 7])
X_test=X_test.reshape([24, 3, 4, 3, 7])
y_train=y_train.reshape([81, 3, 4, 3, 1])
y_val=y_val.reshape([23, 3, 4, 3, 1])
y_test=y_test.reshape([24, 3, 4, 3, 1])
#
#
#
#
# def create_model():
#     model = Sequential()
#     model.add(Conv3D(data_format='channels_last',filters=16, kernel_size=(2,2,2), strides=(1, 1, 1), input_shape=(3, 4, 3, 7),activation='relu'))
#     model.add(Dropout(0.1))
#     model.add(Conv3D(data_format='channels_last',filters=32, kernel_size=(2,2,2), strides=(1, 1, 1),activation='relu'))
#     model.add(Dropout(0.1))
#     model.add(Conv3D(data_format='channels_last', filters=64, kernel_size=(1, 1, 1), strides=(1, 1, 1), activation='relu'))
#     # model.add(Dropout(0.15))
#     model.add(Conv3DTranspose(data_format='channels_last', filters=64, kernel_size=(1, 2, 2), strides=(1, 1, 1)))
#     model.add(Conv3DTranspose(data_format='channels_last',filters=32, kernel_size=(1, 2, 2), strides=(1, 1, 1)))
#     model.add(Conv3DTranspose(data_format='channels_last',filters=16, kernel_size=(1, 1, 1), strides=(1, 1, 1)))
#     model.add(Conv3DTranspose(data_format='channels_last',filters=1, kernel_size=(3, 1, 1), strides=(1, 1, 1)))
#
#     # 打印模型摘要
#     model.summary()
#     # learning_rate_fn = tf.keras.optimizers.schedules.ExponentialDecay(
#     #     0.001)
#     optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
#     model.compile(optimizer = optimizer, loss = "mse", metrics=["mae"])
#     return model
#
#
#
# print("create model and train model")
# model = create_model()
# # # # 定义模型检查点回调函数
# filepath = 'model_Ni.h5'  # 最佳模型文件名#R2: 0.5199410819471817
# checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=100, verbose=1)
# history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=1000,callbacks=[checkpoint])
# # 绘制损失图
# # 设置字体样式为Arial
# matplotlib.rcParams['font.family'] = 'Times New Roman'
# pyplot.plot(history.history['loss'],ls='-',label='train_Loss',c=(0.5, 0.8, 0.9))
# pyplot.plot(history.history['val_loss'],ls='-',label='val_Loss',c=(0.9, 0.5, 0.5))
# pyplot.xlabel('Epoch')
# pyplot.ylabel('Loss')
# # 添加图例
# pyplot.legend()
# pyplot.show()
# # 加载最佳模型
# print('Model evaluation: ', model.evaluate(X_test, y_test))
model=keras.models.load_model('Model/best_model_Cd.h5')
label_pred=model.predict(X_test)
y_test=y_test.reshape(-1,1)
label_pred=label_pred.reshape(-1,1)

true_y=scaler_y.inverse_transform(y_test)
pred_y=scaler_y.inverse_transform(label_pred)
print(true_y.shape)
print(pred_y.shape)
finall_data= np.concatenate((true_y, pred_y), axis=1)
print(finall_data.shape)
finall_data = pd.DataFrame(finall_data)
finall_data.to_excel('Draw/Cd.xlsx', index=False)
R2 = r2_score(y_test, label_pred)
Rmse = np.sqrt((np.mean((y_test-label_pred) ** 2)))
Mae = np.mean(np.abs((y_test-label_pred )** 2))


print('RMSE:', Rmse)
print('MAE:', Mae)
print('R2:',R2)


