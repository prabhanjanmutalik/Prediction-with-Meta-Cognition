import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
import math
import pdb
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)
# fix random seed for reproducibility

np.random.seed(7)
# load the dataset
f=open('lynx.txt')
lynx=f.read()
f.close()
lynx=lynx.split()
lynx=np.reshape(lynx,[114,1])
lynx=np.array(lynx).astype('float32')

'''dataframe = read_csv('lynx.csv', usecols=[1], engine='python', skipfooter=3)
dataset = dataframe.values
dataset = dataset.astype('float32')'''

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(lynx)

x1=np.array(dataset)
p=5
q=2
m0=p+q
m1=100
w1=np.random.random((m0+1,m1))
w2=np.random.random((1,m1+1))
N=len(x1)
yp=np.ones((1,N))
yp=np.reshape(yp,(N,1))
train_size=90
test_size=24
yp_train=np.ones((1,N))
yp_train=np.reshape(yp_train,(N,1))

yp_test=np.ones((1,N))
yp_test=np.reshape(yp_test,(N,1))

net_x2=np.zeros((p,1))
net_x_temp=np.zeros((q,1))
y1=np.zeros((m1,1))
l=0.01
P=np.matrix(np.identity((m1+1)))/l

for k in range(N-1):
    net_x2=np.append(x1[k],net_x2[0:p-1])
    net_x_temp=np.append(net_x_temp[0:q-1],yp[k])
    net_x=np.append(net_x_temp,net_x2)
    net_x=np.append(net_x,1)
    net_x=np.reshape(net_x,(8,1))
    y1=np.tanh(np.transpose(np.matrix(w1)))*np.matrix(net_x)
    y1=np.matrix(np.append(np.array(y1),1))
    y1=np.transpose(y1)
    P=P -(P*y1*np.transpose(y1)*P)/(1+np.transpose(y1)*P*y1)
    yp[k+1]=w2*y1
    alt1=P*y1
    alt2=(x1[k+1]-w2*y1)
    if(k<(train_size-1)):
        yp_train[k+1]=w2*y1
        w2=w2+np.transpose(alt1*alt2)
    else:
        yp_test[k+1]=w2*y1

print(math.sqrt(mean_squared_error(scaler.inverse_transform(x1[train_size+1:len(x1)-1]),scaler.inverse_transform(yp[train_size+2:len(yp)]))))

plt.figure()
plt.plot(scaler.inverse_transform(x1[train_size+1:len(x1)]))
plt.plot(scaler.inverse_transform(yp_test[train_size+2:len(yp)]))
plt.show()
#pdb.set_trace()
#print(error)

