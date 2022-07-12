#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 14:40:02 2022

@author: 1517suj
"""


import random
import os
import shutil

import warnings
warnings.filterwarnings('ignore')
#%%
input_data = '../../../../mask_data'
output_data = '../../../../mask_data_generated'

image_data = sorted(os.listdir(input_data + '/images'))
label_data = sorted(os.listdir(input_data + '/annotations'))

combined_list = list(zip(image_data, label_data))

random.shuffle(combined_list)
split = int(len(combined_list)*0.1)

sub_list1 = combined_list[:split]
sub_list2 = combined_list[split:]


dir_train = '../../../../mask_data_generated/train'
dir_val = '../../../../mask_data_generated/val'

if not os.path.exists(dir_train):
    os.makedirs(dir_train)
if not os.path.exists(dir_val):
    os.makedirs(dir_val)


dst_train = output_data + '/train'
for image, label in sub_list2:
    src_image = input_data + '/images/' + image
    shutil.copy(src_image, dst_train)
    src_label = input_data + '/annotations/' + label 
    shutil.copy(src_label, dst_train)
    
    
dst_val = output_data + '/val'
for image, label in sub_list1:
    src_image = input_data + '/images/' + image
    shutil.copy(src_image, dst_val)
    src_label = input_data + '/annotations/' + label 
    shutil.copy(src_label, dst_val)








