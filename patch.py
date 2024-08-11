import numpy as np



# 读取一个形状为 (708, 108, 9, 7) 的示例数组
point = np.load('data/point.npy')# (708, 108, 9, 7)
Level1=point[:,:,0:3]#(708, 108, 3, 7)
Level2=point[:,:,3:6]#(708, 108, 3, 7)
Level3=point[:,:,6:9]#(708, 108, 3, 7)
Level1_CNN=np.zeros([6372,4,3,3,7])
Level2_CNN=np.zeros([6372,4,3,3,7])
Level3_CNN=np.zeros([6372,4,3,3,7])
for i in range(177):
    for j in range(36):
        Level1_CNN[i*36+j,:,:,:,:]=Level1[0+i*4:4+i*4,0+j*3:3+j*3,:,:]
for i in range(177):
    for j in range(36):
        Level2_CNN[i*36+j,:,:,:,:]=Level2[0+i*4:4+i*4,0+j*3:3+j*3,:,:]
for i in range(177):
    for j in range(36):
        Level3_CNN[i*36+j,:,:,:,:]=Level3[0+i*4:4+i*4,0+j*3:3+j*3,:,:]


print('第一层的形状',Level1_CNN.shape)#(177, 36, 3, 7)
print('第二层的形状',Level2_CNN.shape)#(177, 36, 3, 7)
print('第三层的形状',Level3_CNN.shape)#(177, 36, 3, 7)
np.save('data/level1.npy',Level1_CNN)
np.save('data/level2.npy',Level2_CNN)
np.save('data/level3.npy',Level3_CNN)