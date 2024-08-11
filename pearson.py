import numpy as np
import pandas as pd
Cd=pd.read_excel(open('data/data.xlsx', 'rb'), usecols=[9])
Pb=pd.read_excel(open('data/data.xlsx', 'rb'), usecols=[10])
Cu=pd.read_excel(open('data/data.xlsx', 'rb'), usecols=[11])
Ni=pd.read_excel(open('data/data.xlsx', 'rb'), usecols=[12])
Cd=np.array(Cd)
Pb=np.array(Pb)
Cu=np.array(Cu)
Ni=np.array(Ni)
Cd=Cd.reshape([300])
Pb=Pb.reshape([300])
Cu=Cu.reshape([300])
Ni=Ni.reshape([300])


l12=np.sum((Cd-np.mean(Cd))*(Cu-np.mean(Cu)))/299
l11=np.sqrt(np.sum(np.square(Cd-np.mean(Cd)))/299)
l22=np.sqrt(np.sum(np.square(Cu-np.mean(Cu)))/299)
Pearson=l12/(l11*l22)
print(Pearson)