#!/usr/bin/env python
# coding: utf-8
'''
train up and down model if you have two gpu, otherwise only up or down every time
'''


import argparse, glob
import numpy
import cv2
import chainer
from chainer import cuda
from chainer import serializers
from chainer.functions.loss import mean_absolute_error
import network
import datetime
import imageio
import random

parser = argparse.ArgumentParser(description='')
parser.add_argument('-i', help='Directory path of training data.', default='./training_samples')
parser.add_argument('-o', help='Saved path of an output model file.', default='./models')
parser.add_argument('-l', help='Learning type. (0:downexposure, 1:upexposure)', default='0')
parser.add_argument('-gpu', help='GPU device specifier.', default='-1')
parser.add_argument('-dm', help='File path of a pre-downexposure model.', default='./models/downexposure_model_0.chainer')
parser.add_argument('-um', help='File path of a pre-upexposure model.', default='./models/upexposure_model_0.chainer')
args = parser.parse_args()

gpu = int(args.gpu)
is_upexposure_trained = int(args.l)     
dir_path_list = glob.glob(args.i+'/*')  
dir_path_list = dir_path_list[:]  
if is_upexposure_trained ==0:
        model_path_list = args.dm
else:
    model_path_list = args.um

batch_size = 1       
maximum_epoch = 200                   
predicted_window_len = 8          

lossmask_list = list()
img_shape = (3,512,512)
for i in range(predicted_window_len):
    lossmask = numpy.ones(img_shape[0]*img_shape[1]*img_shape[2]).reshape(img_shape[:1]+(1,)+img_shape[1:])        
    for j in range(predicted_window_len-1,0,-1):
        if i<j:
            append_img = numpy.ones(img_shape[0]*img_shape[1]*img_shape[2]).reshape(img_shape[:1]+(1,)+img_shape[1:])  
        else:
            append_img = numpy.zeros(img_shape[0]*img_shape[1]*img_shape[2]).reshape(img_shape[:1]+(1,)+img_shape[1:])
        lossmask = numpy.hstack([lossmask, append_img])            

    lossmask = numpy.broadcast_to(lossmask, (batch_size,)+lossmask.shape).astype(numpy.float32) 
    lossmask_list.append(lossmask)             

model = network.CNNAE3D512()
optimizer = chainer.optimizers.Adam(alpha=0.0001, beta1=0.5)      
optimizer.setup(model)

xp = cuda.cupy if gpu > -1 else numpy
if gpu>-1:
    cuda.check_cuda_available()
    cuda.get_device(gpu).use()
    model.to_gpu()
    serializers.load_npz(model_path_list, model)
start = datetime.datetime.now()
N = len(dir_path_list)            
for epoch in range(maximum_epoch):
    print ('epoch',epoch)
    start = datetime.datetime.now()
    loss_gen_sum = 0.
    perm = numpy.random.permutation(N)       
    for i in range(N):
        dir_path = dir_path_list[perm[i]]      
        if i%100 == 0:
            print('i',i)
            end = datetime.datetime.now()
        img_path_list_F = glob.glob(dir_path+'/FLDR/*.png')      
        img_path_list_H = glob.glob(dir_path+'/HDR/0.hdr')       
        img_path_list = glob.glob(dir_path+'/LDR/*.png')         
        img_path_list_F.sort()
        img_path_list.sort()
        img_list_F = list()
        img_list_H = list()
        img_list = list()
        if is_upexposure_trained:
            img_order = range(len(img_path_list))
        else:
            img_order = range(len(img_path_list)-1, -1, -1)

        img_H = cv2.imread(img_path_list_H[0], flags=cv2.IMREAD_ANYDEPTH)
        img_H_ = img_H.astype(numpy.float32).transpose(2,0,1)     
        img_list_H.append(img_H_)
        img_list_H = numpy.array(img_list_H)                      

        for j in img_order:
            img_path_F = img_path_list_F[j]
            img_path = img_path_list[j]
            img_F = cv2.imread(img_path_F)
            img = cv2.imread(img_path)
            img_F_ = (img_F.astype(numpy.float32)/255.).transpose(2,0,1)   
            img_ = (img.astype(numpy.float32)/255.).transpose(2,0,1)       
            img_list_F.append(img_F_)
            img_list.append(img_)
        img_list_F = numpy.array(img_list_F)        
        img_list = numpy.array(img_list)       
        for input_frame_id in range(len(img_list)-1):
            start_frame_id = input_frame_id+2            
            end_frame_id = min(start_frame_id+predicted_window_len, len(img_list))
            x_batch = numpy.array([img_list_F[input_frame_id,:,:,:]])        
            y_batch_0 = img_list_H.reshape(x_batch.shape[:2]+(1,)+x_batch.shape[2:]).astype(numpy.float32)    
            y_batch_1 = numpy.array([img_list[start_frame_id:end_frame_id,:,:,:].transpose(1,0,2,3)])       
            y_batch =  numpy.concatenate([y_batch_0, y_batch_1], axis=2)           
            dummy_len = predicted_window_len-y_batch.shape[2]     
            zero_dummy = numpy.zeros(x_batch.size*dummy_len).reshape(y_batch.shape[:2]+(dummy_len,)+y_batch.shape[3:]).astype(numpy.float32) 
            y_batch = numpy.concatenate([y_batch, zero_dummy], axis=2)        
            x_batch = chainer.Variable(xp.array(x_batch))             
            y_batch = chainer.Variable(xp.array(y_batch))             
            lossmask = chainer.Variable(xp.array(lossmask_list[dummy_len]))   
            y_hat = model(x_batch)   
            y_hat = lossmask*y_hat
            loss_gen = mean_absolute_error.mean_absolute_error(y_hat, y_batch)
            model.zerograds()
            loss_gen.backward()
            optimizer.update()
            loss_gen_sum += loss_gen.data*len(x_batch.data)
    print ('loss:',loss_gen_sum/N/(len(img_list)-1))
    end = datetime.datetime.now()
    print('each train time is ',end-start)
    if is_upexposure_trained ==0:
        model_path_list = args.dm
        out_path = args.o+'/downexposure_model_2_'+ str(epoch)+'_.chainer'  
    else:
        model_path_list = args.um
        out_path = args.o+'/upexposure_model_2_'+ str(epoch)+'_.chainer'   

    serializers.save_npz(out_path, model)

