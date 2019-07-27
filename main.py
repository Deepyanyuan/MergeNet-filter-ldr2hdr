#!/usr/bin/env python
# coding: utf-8

import argparse, os, math,glob
import numpy
from PIL import Image
import piexif
import cv2
import chainer
from chainer import cuda
from chainer import serializers
import network
import scipy.io as scio
import datetime

parser = argparse.ArgumentParser(description='')
parser.add_argument('-i', help='File path of input image.', default='./testing_samples')
parser.add_argument('-o', help='Output directory.', default='./results')
parser.add_argument('-gpu', help='GPU device specifier. Two GPU devices must be specified, such as 0,1.', default='0')
parser.add_argument('-dm', help='File path of a downexposure model.', default='./models/downexposure_model_2_18_.chainer')
parser.add_argument('-um', help='File path of a upexposure model.', default='./models/upexposure_model_2_18_.chainer')
parser.add_argument('-al', help='Output directory.', default='0.1')
args = parser.parse_args()

start = datetime.datetime.now()
alpha = numpy.array(args.al).astype(numpy.float32)
dir_path_list = glob.glob(args.i+'/*')
dir_path_list = dir_path_list[:2]
dir_outpath = glob.glob(args.o)
model_path_list = [args.dm, args.um]       
base_outdir_path = args.o       
gpu_list = []
if args.gpu != '-1':
    for gpu_num in (args.gpu).split(','):
        gpu_list.append(int(gpu_num))

'Estimate up-/donwn-exposed images'
model_list = [network.CNNAE3D512(), network.CNNAE3D512()]      
xp = cuda.cupy if len(gpu_list) > 0 else numpy
if len(gpu_list) > 0:
    cuda.check_cuda_available()
    cuda.get_device().use()
    #cuda.get_device(0).use()
    for i in range(2):
        model_list[i].to_gpu()
        serializers.load_npz(model_path_list[i], model_list[i])  
        
else:
    for i in range(2):
        serializers.load_npz(model_path_list[i], model_list[i])  

def estimate_images(input_img, model):      
    # 
    model.train_dropout = False
    input_img_ = (input_img.astype(numpy.float32)/255.).transpose(2,0,1)    
    input_img_ = chainer.Variable(xp.array([input_img_]))
    res  = model(input_img_).data[0]
    if len(gpu_list)>0:
        res = cuda.to_cpu(res)

    out_img_list = list()
    for i in range(res.shape[1]):
        if i ==0:
            out_img = (res[:,i,:,:].transpose(1,2,0)).astype(numpy.float)   
        else:
            out_img = (255.*res[:,i,:,:].transpose(1,2,0)).astype(numpy.uint8)
        out_img_list.append(out_img)

    return out_img_list

print('\nStarting prediction...\n\n')
N = len(dir_path_list)
for i in range (N):
    dir_path = dir_path_list[i]
    frames = [glob.glob(dir_path + '/LDR/1.png')[0], glob.glob(dir_path + '/LDR/4.png')[0], glob.glob(dir_path + '/LDR/7.png')[0]]

    frame_H = [glob.glob(dir_path + '/HDR/1.hdr')[0]]
    HDR_Ground = cv2.imread(frame_H[0], flags=cv2.IMREAD_ANYDEPTH)
    filename_root = os.path.basename(dir_path)    
    print('filename',filename_root)
    save_path = dir_outpath[0] + '/' + filename_root
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    cv2.imwrite(save_path + '/HDR_Ground.hdr', HDR_Ground)

    print('\tReading...')
    for ii in range (len(frames)):
        img = cv2.imread(frames[ii])
        img_zeros = numpy.zeros(numpy.shape(img)).astype(numpy.float)
        out_img_list = list()
        if len(gpu_list)>0:
            cuda.get_device().use()
            for i in range(2):
                out_img_list.extend(estimate_images(img, model_list[i]))
                if i == 0:
                    out_img_list.reverse()  
                    out_img_list.append(img)
        else:
            for i in range(2):
                out_img_list.extend(estimate_images(img, model_list[i]))
                if i == 0:
                    out_img_list.reverse()
                    out_img_list.append(img)
        out_img_list[8] = ((numpy.array(out_img_list[6], dtype=float)+numpy.array(out_img_list[10], dtype=float))/2).astype(numpy.uint8)

        prev_img_log_mean = (out_img_list[7].astype(numpy.float32)+out_img_list[9].astype(numpy.float32))*3-5     
        pre_img_hdr = numpy.power(10, prev_img_log_mean)

        'Select and Merge'
        del out_img_list[9]
        del out_img_list[7]

        threshold = 64           
        stid = 0
        prev_img = out_img_list[7].astype(numpy.float32)       
        out_img_list.reverse()            
        for out_img in out_img_list[8:]:
            img = out_img.astype(numpy.float32)
            if (img>(prev_img+threshold)).sum() > 0:
                break
            prev_img = img[:,:,:]
            stid+=1

        edid = 0
        prev_img = out_img_list[7].astype(numpy.float32)
        out_img_list.reverse()         
        for out_img in out_img_list[8:]:
            img = out_img.astype(numpy.float32)
            if (img<(prev_img-threshold)).sum() > 0:
                break
            prev_img = img[:,:,:]
            edid+=1

        out_img_list_ = out_img_list[7-stid:8+edid]    
        exposure_times = list()
        lowest_exp_time = 1/32.     
        for i in range(len(out_img_list_)):
            exposure_times.append(lowest_exp_time*math.pow(math.sqrt(2.),i))
        exposure_times = numpy.array(exposure_times).astype(numpy.float32)
        print('exposure_times.len',len(exposure_times))
        merge_debvec = cv2.createMergeDebevec()
        hdr_debvec = merge_debvec.process(out_img_list_, times=exposure_times.copy())
        debvec_norm = hdr_debvec/numpy.max(hdr_debvec)*numpy.max(pre_img_hdr)
        merge_final_debvec = alpha*pre_img_hdr+(1-alpha)*debvec_norm
        
        if ii == 0:
            cv2.imwrite(save_path + '/HDR_HybridNet_1.hdr', merge_final_debvec)
            cv2.imwrite(save_path + '/HDR_HybridNet_log_1.hdr', pre_img_hdr)
            cv2.imwrite(save_path + '/HDR_HybridNet_devec_1.hdr', debvec_norm)
        elif ii == 1:
            cv2.imwrite(save_path + '/HDR_HybridNet_4.hdr', merge_final_debvec)
            cv2.imwrite(save_path + '/HDR_HybridNet_log_4.hdr', pre_img_hdr)
            cv2.imwrite(save_path + '/HDR_HybridNet_devec_4.hdr', debvec_norm)
        elif ii == 2:
            cv2.imwrite(save_path + '/HDR_HybridNet_6.hdr', merge_final_debvec)
            cv2.imwrite(save_path + '/HDR_HybridNet_log_6.hdr', pre_img_hdr)
            cv2.imwrite(save_path + '/HDR_HybridNet_devec_6.hdr', debvec_norm)
        # print('\tDone\n')
        del out_img_list

