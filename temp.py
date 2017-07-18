#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 11:41:49 2017

@author: cyrano
"""

import numpy as np
import settings
import helpers
import sys
import os
import glob
import random
import pandas
import ntpath
import cv2
import tqdm
import numpy
from typing import List, Tuple
import math
import shutil
from scipy.misc import toimage
from glob import glob



working_path = settings.BASE_DIR_SSD + "generated_traindata/"

def load_cube_img(src_path, rows, cols, size):
    img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
    # assert rows * size == cube_img.shape[0]
    # assert cols * size == cube_img.shape[1]
    res = numpy.zeros((rows * cols, size, size))

    img_height = size
    img_width = size

    for row in range(rows):
        for col in range(cols):
            src_y = row * img_height
            src_x = col * img_width
            res[row * cols + col] = img[src_y:src_y + img_height, src_x:src_x + img_width]

    return res


def save_cube_img(target_path, cube_img, rows, cols):
    assert rows * cols == cube_img.shape[0]
    img_height = cube_img.shape[1]
    img_width = cube_img.shape[1]
    res_img = numpy.zeros((rows * img_height, cols * img_width), dtype=numpy.uint8)

    for row in range(rows):
        for col in range(cols):
            target_y = row * img_height
            target_x = col * img_width
            res_img[target_y:target_y + img_height, target_x:target_x + img_width] = cube_img[row * cols + col]

    cv2.imwrite(target_path, res_img)

#val_pos = np.load(settings.BASE_DIR_SSD + "project/generated_traindata/val-pos-res.npy")
#
#print(val_pos.shape)
#print(val_pos[10][0].shape)
#print(val_pos[10])
#
#save_cube_img(settings.BASE_DIR_SSD + "project/generated_traindata/val-pos-res-example.png", val_pos[10][0], 8, 8)
    
def merge_val_false_pos_res(): 
    val_false_pos_res = []
    
    neg_samples_list = glob(settings.BASE_DIR_SSD + "generated_traindata/tianchi_val_false_pos/*.png")
    
    print("neg_samples_list count: ", len(neg_samples_list))
    print("neg_samples_list shape: ", (load_cube_img(neg_samples_list[0],6,8,48)).shape)
    
    for i, file_path in enumerate(neg_samples_list):
    #    dia_str = file_path.split('/')[-1].split('_')[-1]
    #    dia = float(dia_str[0:len(dia_str)-4])
        try:
            neg = load_cube_img(file_path,6,8,48)
        except Exception as ex:
            print(file_path)
            continue
        
        val_false_pos_res.append(neg)
    
    np.save(working_path+"val-false-pos.npy", val_false_pos_res)


def merge_train_pos_res(): 
    train_pos_res = []
    
    pos_samples_list = glob(settings.BASE_DIR_SSD + "project/generated_traindata/tianchi_train_pos/*.png")
    
    print("pos_samples_list count: ", len(pos_samples_list))
    print("pos_samples_list shape: ", (load_cube_img(pos_samples_list[0],8,8,64)).shape)
    
    for i, file_path in enumerate(pos_samples_list):
        dia_str = file_path.split('/')[-1].split('_')[-1]
        dia = float(dia_str[0:len(dia_str)-4])
        try:
            pos = load_cube_img(file_path,8,8,64)
        except Exception as ex:
            print(file_path)
            continue
        
        train_pos_res.append((pos, 1, dia))
    
    np.save(working_path+"train-pos-res.npy", train_pos_res)

merge_val_false_pos_res()

train_neg_res = np.load(working_path+"val-false-pos.npy")

print(len(train_neg_res))
print(train_neg_res[0])
print(train_neg_res[0].shape)    
#    


#train_false_pos = []
#for i, file_path in enumerate(false_pos_list):
#    cube = load_cube_img(file_path, 6, 8, 48)
#    train_false_pos.append(cube)
#    
#np.save(working_path+"val_false_pos.npy", train_false_pos)
#
#print("false_pos_list count: ", len(train_false_pos))
#print("false_pos_list shape: ", train_false_pos[0].shape)

#edges = np.load(working_path+"train-neg-edge.npy")
#
#print(edges.shape)
#b = np.nonzero(edges)
#print(np.array(b).ndim)
#print(b)


#train_data_path =  glob(settings.BASE_DIR_SSD + "project/generated_traindata/*.npy")
#
#for i, path in enumerate(train_data_path):
#    print(path)
#    
#    file = np.load(path)
#    
#    print(file.shape)

#TIANCHI_WORK_DIR = "test2_subset0"
#
#all_mhd = []
#    
#for subject_no in range(settings.TIANCHI_SUBSET_START_INDEX, settings.TIANCHI_SUBSET_END_INDEX):
#    mhd_dir = settings.TIANCHI_RAW_SRC_DIR + TIANCHI_WORK_DIR + str(subject_no) + "/"
#    mhd_paths = glob(mhd_dir + "*.mhd")
#    all_mhd.extend(mhd_paths)
#    
#print("Total mhd: ", len(all_mhd))
#
#rows = []
#
#for i, path in enumerate(all_mhd):
#    patient_id = path.split('/')[-1].replace(".mhd", "")
#    rows.append(patient_id)
#    
#df = pandas.DataFrame(rows, columns=["patient_id"])
#df.to_csv(settings.EXTRA_DATA_DIR + "test2/seriesuids.csv", index=False)    
