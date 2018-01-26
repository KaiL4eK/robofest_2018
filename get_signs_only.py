from __future__ import print_function

import os
import numpy as np
import cv2

raw_path  = ['data/positive_bbox_class' ]

def print_process(index, total, step):
    if index % step == 0:
        print('Done: {0}/{1} images'.format(index, total))

names = []

def create_train_data():
    total = 0
    i = 0
    
    avg_imgs_width  = 0
    avg_imgs_height = 0

    print('-'*30)
    print('Creating training images...')
    print('-'*30)

    dirpath = raw_path[0]
    print('Processing path: {}'.format(dirpath))
    images = os.listdir(dirpath)
    imgs_count = len(images)

    total += imgs_count

    for image_name in images:
        info   = image_name.split(';')
        name   = info[5].split('.')[0]

        if name not in names:
            names.append(name)

    sign_idxs = [0] * len(names)
    print(names)
    print(sign_idxs)

    if not os.path.exists("data/signs_only"):
        os.mkdir('data/signs_only')

        for name in names:
            os.mkdir('data/signs_only/%s' % name)

    for image_name in images:
        # Save image
        # print(image_name)
        img = cv2.imread(os.path.join(dirpath, image_name))
        if img is None:
            continue

        img_height, img_width, channels = img.shape

        # Get info about bbox
        info   = image_name.split(';')
        ul_x   = max(0, int(info[1]))
        ul_y   = max(0, int(info[2]))
        width  = max(0, int(info[3]))
        height = max(0, int(info[4]))
        name   = info[5].split('.')[0]

        idx = sign_idxs[names.index(name)] 
        sign_idxs[names.index(name)] += 1

        lr_x = min(ul_x + width, img_width)
        lr_y = min(ul_y + height, img_height)

        image_sign = img[ul_y:lr_y, ul_x:lr_x]

        avg_imgs_width  += lr_x - ul_x
        avg_imgs_height += lr_y - ul_y

        # image_sign = cv2.resize(image_sign, (64, 64))

        cv2.imwrite( 'data/signs_only/%s/%d.png' % (name, idx), image_sign )

        i += 1
        print_process(i, total, 20)
        
    avg_imgs_width  /= float(imgs_count)
    avg_imgs_height /= float(imgs_count)

    print(avg_imgs_width, avg_imgs_height)
    print('{} / Sum: {}'.format(sign_idxs, sum(sign_idxs)))

if __name__ == '__main__':
    create_train_data()