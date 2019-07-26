# coding:utf-8

import numpy
import OpenEXR
import Imath
import imageio
import glob
import os
import cv2
import scipy.io as scio


def ext2hdr(exrpath):
    File = OpenEXR.InputFile(exrpath)
    PixType = Imath.PixelType(Imath.PixelType.FLOAT)
    DW = File.header()['dataWindow']
    Size = (DW.max.x - DW.min.x + 1, DW.max.y - DW.min.y + 1)
    rgb = [numpy.fromstring(File.channel(c, PixType), dtype=numpy.float32) for c in 'RGB']
    r =numpy.reshape(rgb[0],(Size[1],Size[0]))
    g =numpy.reshape(rgb[1],(Size[1],Size[0]))
    b =numpy.reshape(rgb[2],(Size[1],Size[0]))
    hdr = numpy.zeros((Size[1],Size[0],3),dtype=numpy.float32)
    hdr[:,:,0] = b
    hdr[:,:,1] = g
    hdr[:,:,2] = r
    return hdr

dir_path_list = glob.glob('./exr/*')
dir_path_list = dir_path_list[:3]
N = len(dir_path_list)
for i in range(N):
    dir_path = dir_path_list[i]
    files = glob.glob(dir_path+'/*.exr')
    #savepath = glob.glob(dir_path+'/HDR_format')
    savepath = dir_path+'./HDR_format'
    for file in files:
        hdr = ext2hdr(file)
        filename,file_ext = os.path.splitext(file)   
        filename = os.path.basename(filename)     
        filename = filename + '.hdr'   
        curpath = os.path.join(savepath,filename) 
        cv2.imwrite(curpath,hdr)
        #imageio.imwrite(curpath,hdr)

        hdr_o = cv2.imread(curpath, flags= cv2.IMREAD_ANYDEPTH)       
        scio.savemat(filename+'_exr.mat',{'exr':hdr})
        scio.savemat(filename+'_hdr.mat',{'hdr':hdr_o})
        
print('u are so smart!')
        

