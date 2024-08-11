import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn import model_selection


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

print(dataset.shape,labelset.shape)


dataset = np.transpose(dataset, (1, 0, 2))
labelset = np.transpose(labelset, (1, 0, 2))
print("数据形状:", dataset.shape)
print("标签形状:", labelset.shape)
# # # #3DCNN的输入为（4, 4, 3, 7）,4：每次卷积4个深度，4，3：每层的点位数，7：特征数
# 打印合并后的数组形状
#将数据合并为3DCNN所需的格式
data=np.zeros([128,3,4,3,7])
label=np.zeros([128,3,4,3,1])

for i in range(64):
    data[i,:,:,:,:]=dataset[0:3,i:i+12,:].reshape([3,4,3,7])
    label[i, :, :, :] = labelset[0:3,i:i+12,:].reshape([3,4,3,1])
for i in range(64,128):
    data[i,:,:,:,:]=dataset[1:4,i-64:i-64+12,:].reshape([3,4,3,7])
    label[i, :, :, :] = labelset[1:4,i-64:i-64+12,:].reshape([3,4,3,1])
#
print("数据形状:", data.shape)
print("标签形状:", label.shape)

X_train, X_val, y_train, y_val = model_selection.train_test_split(data, label, test_size = 0.36, random_state = 1234)
X_val, X_test, y_val, y_test = model_selection.train_test_split(X_val, y_val, test_size = 0.5, random_state = 1234)


print("训练集数据形状:", X_train.shape)
print("训练集标签形状:", y_train.shape)
print("验证集数据形状:", X_val.shape)
print("验证集标签形状:", y_val.shape)
print("测试集数据形状:", X_test.shape)
print("测试集标签形状:", y_test.shape)
X_train=X_train.reshape([2916,7])
X_val=X_val.reshape([828,7])
X_test=X_test.reshape([864,7])
y_train=y_train.reshape([2916,1])
y_val=y_val.reshape([828,1])
y_test=y_test.reshape([864,1])



train_data= np.concatenate((X_train, y_train), axis=1)#(229392, 11)
val_data= np.concatenate((X_val, y_val), axis=1)#(229392, 11)
test_data= np.concatenate((X_test, y_test), axis=1)#(229392, 11)


df = pd.DataFrame(train_data)
df.to_excel('X_train.xlsx', index=False)
df = pd.DataFrame(val_data)
df.to_excel('X_val.xlsx', index=False)
df = pd.DataFrame(test_data)
df.to_excel('X_test.xlsx', index=False)
