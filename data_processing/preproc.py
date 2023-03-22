import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from PIL import Image
import tifffile
import json
import torchvision.transforms as transforms

data_dir = 'data'

image_dir = os.path.join(data_dir,'train_images')
label_dir = os.path.join(data_dir, 'train_annotation')
meta_path = os.path.join(data_dir,'train.csv')
mask_dir = os.path.join(data_dir, 'train_mask')
resized_dir = os.path.join(data_dir,'resized_images')

meta_df = pd.read_csv(meta_path).sort_values(by = 'id')



def resizer(im_name,shape):
    '''
    The resizer takes in 2 args:
    im_name- name of the image
    scale- percentage by which the image has to be reduced
    '''

    image_path = os.path.join(image_dir, im_name +'.tiff')
    im_read = tifffile.imread(image_path)
    if shape is None:
        shape = im_read.shape[:2]
    print('File name: {}, original size: {}, resized to: {}'.format(im_name , (im_read.shape[0], im_read.shape[1]), shape))
    resized = cv2.resize(im_read, shape, interpolation=cv2.INTER_AREA)
    image_path = os.path.join(resized_dir, ('r_' + im_name +'.tiff'))
    tifffile.imwrite(image_path, resized)



def rle2mask(rle,shape):
    ## to see a complete code breakdown, REFER version 3.
    s = rle.split()
    # the "s" here is of dtype('<U7') hence we convert it to "int"
    # very very important step 
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T

def resize_mask(im_name,shape):
    '''
    reads RLE encodings from the df
    converts to masks and resizes it to a scaling_percentage of original size
    '''
    im_read = tifffile.imread(os.path.join(image_dir, im_name +'.tiff'))
    if shape is None:
        shape = im_read.shape[:2]
    mask_rle = meta_df[meta_df["id"] == int(im_name)]["rle"].values[0]
    mask = rle2mask(mask_rle, (im_read.shape[1], im_read.shape[0]))*255
    print('File name: {}, original size: {}, resized to: {}'.format(im_name, (im_read.shape[0], im_read.shape[1]), shape))
    resized = cv2.resize(mask, shape, interpolation=cv2.INTER_AREA)
    image_path = os.path.join(mask_dir, (im_name + '.tiff'))
    tifffile.imwrite(image_path, resized) 

#Mettre None pour ne pas resize
shape = None

for image in os.listdir(image_dir):
    image_name = image[:-5]
    resizer(image_name,shape)           #Resize l'image et l'enregistre dans resize_images (avec r_ en préfixe)
    resize_mask(image_name,shape)       #Resize et enregistre les masques : tableaux binaires*255
