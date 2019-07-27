# coding:utf-8
'''
Generate training data pairs: hdr2ldr
including one log-domain HDR image, one original-domain HDR image, 
9 traditional multi-exposure LDR images, 9 filtered multi-exposure LDR images
'''
import numpy as np
import cv2
import glob, argparse, math
import OpenEXR
import Imath
import imageio
import os, sys
import datetime
import random

parser = argparse.ArgumentParser(description='')
parser.add_argument('-i', help='Directory path of hdr images.', default='./hdr')                          
parser.add_argument('-o', help='Directory path of ldr images.', default='./training_samples_2')           

args = parser.parse_args()

# definite camera response curve function   
def func_0(x):
    result = 0.02075*np.power(x, 3) + 0.5034 * np.power(x, 2) + 0.4727 * x - 0.001136
    result[result>1.0]=1.0 
    result[result<0.0]=0.0 
    return result
def func_1(x):
    result = 0.9491*np.power(x, 3) - 2.97 * np.power(x, 2) + 3.114 * x - 0.1031
    result[result>1.0]=1.0  
    result[result<0.0]=0.0 
    return result
def func_2(x):
    result = 0.2108*np.power(x, 3) -0.9448 * np.power(x, 2) + 1.711 * x +0.0246
    result[result>1.0]=1.0   
    result[result<0.0]=0.0 
    return result
def func_3(x):
    result = 2.909*np.power(x, 3) -5.858 * np.power(x, 2) + 3.908 * x +0.0883
    result[result>1.0]=1.0   
    result[result<0.0]=0.0 
    return result
def func_4(x):
    result = 1.462*np.power(x, 3) - 3.16 * np.power(x, 2) + 2.618 * x +0.1047
    result[result>1.0]=1.0
    result[result<0.0]=0.0 
    return result
func_dict = {'mark0': func_0, 'mark1': func_1, 'mark2': func_2, 'mark3': func_3, 'mark4': func_4}
mark_list = ['mark0', 'mark1', 'mark2', 'mark3', 'mark4']                   # mark_list 是指相机函数标志物

# define digital filter function
def hdr_filter_func(hdr):
    height, width, channel = np.shape(hdr)
    temp = np.zeros(height * width * channel).reshape(height, width, channel)
    temp[:,:,0] = hdr[:,:,0] * 0.6
    temp[:,:,1] = hdr[:,:,1] * 0.9
    temp[:,:,2] = hdr[:,:,2] * 0.3
    return temp

# exposure time function
def exposure_times(tao, T):
    delt_t = list()
    tt = int(T/2+1)
    for t in range(tt):
        delt_t_ = math.pow(1/tao, t)
        delt_t.append(delt_t_)
    delt_t.reverse()
    for t in range(tt-1):
        delt_t_ =math.pow(tao,t+1)
        delt_t.append(delt_t_)
    delt_t = np.array(delt_t)
    return delt_t

tao = math.sqrt(2)                                                         
T = 8                                                                      
normal_value = 3                                                           
dir_in_path_list = glob.glob(args.i+'/*')
dir_in_path_list = dir_in_path_list[:]                                      
dir_out_path = glob.glob(args.o)
Times = exposure_times(tao,T)                                               

start = datetime.datetime.now ()
N = len(dir_in_path_list)
for i in range(N):
    dir_in_path = dir_in_path_list[i]                                      
    filename_root = os.path.basename(dir_in_path)                          
    files_hdr_path_list = glob.glob(dir_in_path+'/*.hdr')
    for file_num, file in enumerate (files_hdr_path_list):
        if file_num % 1 == 0:
            hdr = cv2.imread(file, flags=cv2.IMREAD_ANYDEPTH)          # read HDR dataset      
            hdr_0 = hdr + (10**-8)                                         
            filename_hdr, file_format = os.path.splitext(file)             
            filename_sub = os.path.basename(filename_hdr)                  
            print('file name:', filename_sub)
            hdr_log = np.log10(hdr_0)
            hdr_log_norm = (hdr_log+5)/6.0                                 
            hdr_mean = np.mean(hdr_0)
            hdr_norm = hdr_0/(normal_value * hdr_mean)                     
            hdr_filter = hdr_filter_func(hdr_norm)                         
            hdr_norm_exposure = list()
            hdr_filter_exposure = list()
            for i in range(T+1):
                Time = Times[i]
                hdr_norm_exposure.append(hdr_norm * Time)
                hdr_filter_exposure.append(hdr_filter * Time)
            hdr_norm_exposure = np.array(hdr_norm_exposure)
            hdr_filter_exposure = np.array(hdr_filter_exposure)

            for i in range(5):
                mark = mark_list[i]
                ldr_norm_temp = func_dict[mark](hdr_norm_exposure)
                ldr_filter_temp = func_dict[mark](hdr_filter_exposure)
                save_root_path = dir_out_path[0] + '/' + filename_root + '_' + filename_sub + '_' + mark + '_sub'
                
                count = 0
                image_each = 3
                exposure_N, height, width, channel = np.shape(ldr_norm_temp)
                img_patch = np.min([height, width])

                if img_patch > 1023:  
                    while count < image_each:
                        width1 = random.randint(0, width - img_patch)
                        height1 = random.randint(0, height - img_patch)
                        width2 = width1 + img_patch
                        height2 = height + img_patch
                        cut_hdr_temp_0 = hdr_log_norm[height1:height2, width1:width2, :]                # split the log-domain HDR images, normalization
                        cut_hdr_temp_1 = hdr_0[height1:height2, width1:width2, :]                       # split the original-domain HDR images
                        cut_ldr_temp_0 = ldr_norm_temp[:, height1:height2, width1:width2, :]            # generate the traditonal LDR images, normalization
                        cut_ldr_temp_1 = ldr_filter_temp[:, height1:height2, width1:width2, :]          # generate the filtered LDR images, normalization

                        re_size = (512, 512)
                        shrink_cut_hdr_temp_0 = cv2.resize(cut_hdr_temp_0, re_size, interpolation=cv2.INTER_AREA)
                        shrink_cut_hdr_temp_1 = cv2.resize(cut_hdr_temp_1, re_size, interpolation=cv2.INTER_AREA)

                        num_str = str(count + 1).rjust(2, '0')
                        savepath = save_root_path + num_str
                        class_H_path = savepath + '/HDR'
                        class_L_path = savepath + '/LDR'
                        class_F_path = savepath + '/FLDR'
                        os.makedirs(class_H_path)
                        os.makedirs(class_L_path)
                        os.makedirs(class_F_path)
                        cv2.imwrite(class_H_path + '/0.hdr', shrink_cut_hdr_temp_0)                       # save the log-domain HDR images as ground-truth
                        cv2.imwrite(class_H_path + '/1.hdr', shrink_cut_hdr_temp_1)                       # save the original-domain HDR images as ground-truth
                        for n in range(exposure_N):
                            shrink_cut_ldr_temp_0 = cv2.resize(cut_ldr_temp_0[n] * 255, re_size, interpolation=cv2.INTER_AREA)
                            shrink_cut_ldr_temp_1 = cv2.resize(cut_ldr_temp_1[n] * 255, re_size, interpolation=cv2.INTER_AREA)
                            cv2.imwrite(class_L_path + '/' + str(n) + '.png', shrink_cut_ldr_temp_0)      # save the traditonal LDR images as ground-truth          
                            cv2.imwrite(class_F_path + '/' + str(n) + '.png', shrink_cut_ldr_temp_1)      # save the filtered LDR images as ground-truth           
                        count += 1

                else:
                    re_size = (512, 512)
                    shrink_cut_hdr_temp_0 = cv2.resize(hdr_log_norm, re_size, interpolation=cv2.INTER_AREA)
                    shrink_cut_hdr_temp_1 = cv2.resize(hdr_0, re_size, interpolation=cv2.INTER_AREA)
                    cut_ldr_temp_0 = ldr_norm_temp
                    cut_ldr_temp_1 = ldr_filter_temp

                    num_str = str(count + 1).rjust(2, '0')
                    savepath = save_root_path + num_str
                    class_H_path = savepath + '/HDR'
                    class_L_path = savepath + '/LDR'
                    class_F_path = savepath + '/FLDR'
                    os.makedirs(class_H_path)
                    os.makedirs(class_L_path)
                    os.makedirs(class_F_path)
                    cv2.imwrite(class_H_path + '/0.hdr', shrink_cut_hdr_temp_0)                      
                    cv2.imwrite(class_H_path + '/1.hdr', shrink_cut_hdr_temp_1)                      
                    for n in range(exposure_N):
                        shrink_cut_ldr_temp_0 = cv2.resize(cut_ldr_temp_0[n] * 255, re_size, interpolation=cv2.INTER_AREA)
                        shrink_cut_ldr_temp_1 = cv2.resize(cut_ldr_temp_1[n] * 255, re_size, interpolation=cv2.INTER_AREA)
                        cv2.imwrite(class_L_path + '/' + str(n) + '.png', shrink_cut_ldr_temp_0)               
                        cv2.imwrite(class_F_path + '/' + str(n) + '.png', shrink_cut_ldr_temp_1)               

end = datetime.datetime.now()
print(end-start)
print('success!')