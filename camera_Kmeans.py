# coding:utf-8
'''
Select and plot camera response curves
'''
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import scipy.io as scio
import math

filename = './camera_function'
brightness = scio.loadmat(filename+'/brightness.mat')['brightness']
irradiance = scio.loadmat(filename+'/irradiance.mat')['irradiance']

brightness = np.array(brightness)
irradiance = np.array(irradiance)             
height, width = np.shape(brightness)          

data = brightness.transpose()     


label_pred_ = scio.loadmat('./camera_function/label_pred.mat')['label_pred']
centroids = scio.loadmat('./camera_function/centroids.mat')['centroids']
label_pred = np.array(label_pred_).transpose()
print(np.shape(label_pred))
print(np.shape(centroids))
print(label_pred_)

for i in range(5):
    if i ==0:
        plt.plot(irradiance, centroids[i], '#e24fff')
    if i ==1:
        plt.plot(irradiance, centroids[i], 'g')
    if i ==2:
        plt.plot(irradiance, centroids[i], 'r')
    if i ==3:
        plt.plot(irradiance, centroids[i], 'k')
    if i ==4:
        plt.plot(irradiance, centroids[i], 'c')

# fit
irradiance_ = irradiance.transpose()

temp_0 =  centroids[0]
f_0 = np.polyfit(irradiance_[0], temp_0, 3)
p_0 = np.poly1d(f_0)
print('p_0',p_0)
yval_0 = p_0(irradiance)
plt.plot(irradiance, yval_0, '#A0522D')

temp_1 =  centroids[1]
f_1 = np.polyfit(irradiance_[0], temp_1, 3)
p_1 = np.poly1d(f_1)
print('p_1',p_1)
yval_1 = p_1(irradiance)
plt.plot(irradiance, yval_1, '#A0522D')

temp_2 =  centroids[2]
f_2 = np.polyfit(irradiance_[0], temp_2, 3)
p_2 = np.poly1d(f_2)
print('p_2',p_2)
yval_2 = p_2(irradiance)
plt.plot(irradiance, yval_2, '#e24fff')

temp_3 =  centroids[3]
f_3 = np.polyfit(irradiance_[0], temp_3, 3)
p_3 = np.poly1d(f_3)
print('p_3',p_3)
yval_3 = p_3(irradiance)
plt.plot(irradiance, yval_3, '#e24fff')

temp_4 =  centroids[4]
f_4 = np.polyfit(irradiance_[0], temp_4, 3)
p_4 = np.poly1d(f_4)
print('p_4',p_4)
yval_4 = p_4(irradiance)
plt.plot(irradiance, yval_4, '#e24fff')


plt.show()

