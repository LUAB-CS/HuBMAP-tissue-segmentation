import os
import numpy as np
import pandas as pd
import cv2
import tifffile


def resize(image_path,shape,resized_dir):
    '''
    The resizer takes in 2 args:
    im_name- name of the image
    scale- percentage by which the image has to be reduced
    '''
    im_read = tifffile.imread(image_path)
    im_name = image_path.split('/')[0][:-5]
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

#def resize_mask(image_path,shape,resized_dir):
#    '''
#    reads RLE encodings from the df
#    converts to masks and resizes it to a scaling_percentage of original size
#    '''
#    im_read = tifffile.imread(image_path)
#    im_name = image_path.split('/')[0][:-5]
#    if shape is None:
#        shape = im_read.shape[:2]
#    mask_rle = meta_df[meta_df["id"] == int(im_name)]["rle"].values[0]
#    mask = rle2mask(mask_rle, (im_read.shape[1], im_read.shape[0]))*255
#    print('File name: {}, original size: {}, resized to: {}'.format(im_name, (im_read.shape[0], im_read.shape[1]), shape))
#    resized = cv2.resize(mask, shape, interpolation=cv2.INTER_AREA)
#    image_path = os.path.join(mask_dir, (im_name + '.tiff'))
#    tifffile.imwrite(image_path, resized)


def create_masks_as_tiff(data_dir):
    meta_df = pd.read_csv(os.path.join(data_dir,'train.csv'))
    if 'train_mask' not in os.listdir(data_dir):
        os.mkdir(os.path.join(data_dir, 'train_masks'))

    for image in os.listdir(os.path.join(data_dir,'train_images')):
        image_metadata = meta_df[meta_df["id"] == int(image[:-5])]
        mask_rle = image_metadata["rle"].values[0]
        shape = (image_metadata['img_height'].values[0],image_metadata['img_width'].values[0])
        mask_array = rle2mask(mask_rle, shape)*255
        image_path = os.path.join(os.path.join(data_dir, 'train_masks'), image)
        tifffile.imwrite(image_path, mask_array)

def preprocess_images_and_masks(data_dir,reshape_size=None):

    image_dir = os.path.join(data_dir,'train_images')
    mask_dir = os.path.join(data_dir, 'train_masks')
    resized_images_dir = os.path.join(data_dir,'resized_train_images')
    resized_masks_dir = os.path.join(data_dir,'resized_train_masks')
    if 'resized_train_images' not in os.listdir(data_dir):
        os.mkdir(os.path.join(data_dir, 'resized_train_images'))
    if 'resized_train_images' not in os.listdir(data_dir):
        os.mkdir(os.path.join(data_dir, 'resized_train_images'))

    if reshape_size:
        for image in os.listdir(image_dir):
            image_path = os.path.join(image_dir, image)
            resize(image_path,reshape_size,resized_images_dir)           #Resize l'image et l'enregistre dans resize_images (avec r_ en préfixe)

        for image in os.listdir(mask_dir):
            image_path = os.path.join(mask_dir, image)
            resize(image_path,reshape_size,resized_masks_dir)
