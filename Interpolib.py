from sklearn.metrics import r2_score
from sklearn.svm import SVR
import numpy as np
import pandas as pd
from matplotlib import pyplot
from sklearn import model_selection
from pykrige.ok3d import OrdinaryKriging3D

#读取每个层的钻孔数据
data_1=pd.read_excel(open('data/data.xlsx', 'rb'), usecols=[1,2,3])
Y_1=pd.read_excel(open('data/data.xlsx', 'rb'), usecols=[9])
Y_2=pd.read_excel(open('data/data.xlsx', 'rb'), usecols=[10])
Y_3=pd.read_excel(open('data/data.xlsx', 'rb'), usecols=[11])
Y_4=pd.read_excel(open('data/data.xlsx', 'rb'), usecols=[12])


data_1=np.array(data_1)
Y_1=np.array(Y_1)
Y_2=np.array(Y_2)
Y_3=np.array(Y_3)
Y_4=np.array(Y_4)



print(data_1.shape)
print(Y_1.shape)
print(Y_2.shape)
print(Y_3.shape)
print(Y_4.shape)



# ok3d = OrdinaryKriging3D(train_data[:, 0], train_data[:, 1], train_data[:, 2], train_data[:, 3], variogram_model="linear",verbose=1)
ok3d = OrdinaryKriging3D(data_1[:, 0], data_1[:, 1], data_1[:, 2], Y_3[:, 0], variogram_model="spherical",verbose=1)
# variogram_model（str，可选） - 指定要使用的变异函数模型; 可能是以下之一：线性，幂，高斯，球形，指数，孔效应。 默认是线性变异函数模型。


X_test=pd.read_excel(open('data/data.xlsx', 'rb'), usecols=[1,2,3],sheet_name='Sheet2')
X_test=np.array(X_test)
#预测
y_pred, ss3d = ok3d.execute("points", X_test[:, 0], X_test[:, 1], X_test[:, 2])


for i in range(X_test.shape[0]):
    print(y_pred[i])

