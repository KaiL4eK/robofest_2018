from __future__ import print_function

import os
import numpy as np
import cv2
import glob
import zipfile

files_base_path  = '../data/positive_bbox_class'

data_path = 'npy'

npy_img_height = 240
npy_img_width = 320

# Bbox is compiled as [ul_x, ul_y, w, h]
check_data = False

def print_process(index, total):
    if index % 100 == 0:
        print('Done: {0}/{1} images'.format(index, total))

stats = {
            'forward and right' : 0,
            'forward'           : 0,
            'right'             : 0,
            'left'              : 0,
            'brick'             : 0
        }

def create_train_data():
    total = 0
    i = 0

    print('-'*30)
    print('Creating training images...')
    print('-'*30)

    # for filename in files_base_path:
    images = os.listdir(files_base_path)
    ctx_files = [f for f in images if f.endswith('.ctx')]

    for file in ctx_files:
        info   = file.split(';')
        # ul_x   = max(0, int(float(info[1]) * scale_x))
        # ul_y   = max(0, int(float(info[2]) * scale_y))
        # width  = max(0, int(float(info[3]) * scale_x))
        # height = max(0, int(float(info[4]) * scale_y))
        name   = info[5].split('.')[0]

        stats[name] += 1

        if name == 'brick':
            continue

        total += 1

    # total += len(ctx_files)

    imgs        = np.ndarray((total, npy_img_height, npy_img_width, 3), dtype=np.uint8)
    imgs_mask   = np.ndarray((total, npy_img_height, npy_img_width), dtype=np.uint8)

    print(stats)

    for file in ctx_files:
        info   = file.split(';')
        # ul_x   = max(0, int(float(info[1]) * scale_x))
        # ul_y   = max(0, int(float(info[2]) * scale_y))
        # width  = max(0, int(float(info[3]) * scale_x))
        # height = max(0, int(float(info[4]) * scale_y))
        name   = info[5].split('.')[0]
        if name == 'brick':
            continue

        archive = zipfile.ZipFile(os.path.join(files_base_path, file), 'r')

        img_file  = archive.read('image.png')
        mask_file = archive.read('mask.png')
        

        img  = cv2.imdecode(np.frombuffer(img_file, np.uint8), 1)
        mask = cv2.imdecode(np.frombuffer(mask_file, np.uint8), 0)  

        img  = cv2.resize(img, (npy_img_width, npy_img_height), interpolation = cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (npy_img_width, npy_img_height), interpolation = cv2.INTER_NEAREST)

        # Render for control
        # _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.drawContours(img, contours, -1, (255, 255, 0), 4)

        # cv2.imshow('1', np.hstack((img, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))))
        # cv2.waitKey(0)

        imgs[i]         = img
        imgs_mask[i]    = mask

        print_process(i, total)
        i += 1

    np.save(data_path + '/imgs_train.npy', imgs)
    np.save(data_path + '/imgs_mask_train.npy', imgs_mask)

    print('Loading done.')
    print('Saving to .npy files done.')

def npy_data_load_images():
    return np.load(data_path + '/imgs_train.npy')

def npy_data_load_masks():
    return np.load(data_path + '/imgs_mask_train.npy')

if __name__ == '__main__':
    create_train_data()