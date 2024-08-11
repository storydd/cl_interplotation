import random
import numpy as np
i=0
j=0
point=np.zeros([708,108,9,7])
tezheng=np.zeros([7])
for x in np.arange(4004058.0,4005119.0,1.5):
    j = 0
    for y in np.arange(40540313.0,40540474.0,1.5):
        for z in np.arange(0,4.5,0.5):
            if (z<=1.5):
                tezheng = [x, y, z, random.uniform(7.5, 9), 10.000, 15.000, 0.700]
                point[i, j, int(z/0.5),:]=tezheng
                # print(point[i, j, int(z/0.5),:])
            if (z>1.5) & (z<=3.0):
                tezheng = [x, y, z, random.uniform(7.5, 9), 10.000, 10.000, 0.500]
                point[i, j, int(z/0.5), :] = tezheng
                # print(point[i, j, int(z / 0.5), :])
            if (z>3.0) & (z<=3.5):
                tezheng = [x, y, z, random.uniform(7.5, 9), 0.001,4.000,0.200]
                point[i, j, int(z/0.5), :] = tezheng
                # print(point[i, j, int(z/0.5),:])
            if (z>3.5):
                tezheng = [x, y, z, random.uniform(7.5, 9), 0.200,2.000,0.000]
                point[i, j, int(z/0.5), :] = tezheng
                # print(point[i, j, int(z/0.5),:])
        print(i, j)
        j = j + 1
    i = i + 1
np.save('data/point.npy',point)